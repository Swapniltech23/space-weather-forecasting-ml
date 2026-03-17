# space_weather_dashboard.py  — +3h ahead, no refit, 121-feature schema
import json, time, io, re
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os


# ========================= CONFIG =========================
MODEL_PATH = "model/model.json"
SCALER_PATH = "model/scaler.joblib"
NASA_KEY = os.getenv("NASA_KEY") 
HORIZON_H   = 3              # <-- fixed at +3h (matches your trained model)

st.set_page_config(page_title=f"Space Weather — Dst (+{HORIZON_H}h)",
                   page_icon="🛰️", layout="wide")
st.title(f"🛰️ Live Space Weather Forecast — Dst (+{HORIZON_H} h)")

# ========================= Utility helpers =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    mapping = {}
    for c in df.columns:
        x = c.strip().lower().replace(" ", "_").replace("-", "_")
        x = x.replace("(", "").replace(")", "")
        if x in {"bz","bz_gsm","bzgsm","bz_gse"}: mapping[c] = "Bz_GSM"
        elif x in {"density","np","proton_density"}: mapping[c] = "Density"
        elif x in {"speed","v","v_sw","solar_wind_speed"}: mapping[c] = "Speed"
        elif x in {"flow_pressure","dynamic_pressure","pressure","p_dyn"}: mapping[c] = "Flow_Pressure"
        elif x in {"kp","kp_index"}: mapping[c] = "Kp"
        else: mapping[c] = c
    return df.rename(columns=mapping)

def to_numeric_all(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _safe_swpc_json(url: str, retries: int = 3, timeout: int = 20):
    """SWPC JSON sometimes has extra chars; trim to bracket block if needed."""
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                txt = r.text.strip()
                s, e = txt.find("["), txt.rfind("]")
                if s != -1 and e != -1 and e > s:
                    return json.loads(txt[s:e+1])
        except Exception:
            time.sleep(1)
    return []

def _parse_swpc_json(json_obj) -> pd.DataFrame:
    if not json_obj or len(json_obj) < 2:
        return pd.DataFrame()
    header = json_obj[0]
    body   = json_obj[1:]
    df = pd.DataFrame(body, columns=header)
    if "time_tag" in df.columns:
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
        df = df.dropna(subset=["time_tag"]).set_index("time_tag")
    df = df.drop(columns=[c for c in df.columns if c.lower() == "time_tag"], errors="ignore")
    return df

# ========================= Live feeds =========================
def fetch_kp_hourly() -> pd.DataFrame:
    raw = _safe_swpc_json("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
    kp  = _parse_swpc_json(raw)
    if "Kp" in kp.columns:
        kp["Kp"] = pd.to_numeric(kp["Kp"], errors="coerce")
        kp = kp[["Kp"]]
    else:
        kp = pd.DataFrame(index=kp.index, data={"Kp": np.nan})
    return kp.resample("H").mean(numeric_only=True)

def fetch_solar_wind_hourly() -> pd.DataFrame:
    p_raw = _safe_swpc_json("https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json")
    m_raw = _safe_swpc_json("https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json")
    p = _parse_swpc_json(p_raw)
    m = _parse_swpc_json(m_raw)

    p_keep = [c for c in p.columns if c.lower() in {"density","speed","temperature"}]
    m_keep = [c for c in m.columns if c.lower() in {"bz_gse","bz_gsm","bt_gse","bt"}]
    p = p[p_keep] if p_keep else pd.DataFrame(index=p.index)
    m = m[m_keep] if m_keep else pd.DataFrame(index=m.index)

    sw = p.join(m, how="outer")
    sw = normalize_cols(sw)
    sw = to_numeric_all(sw)

    if "Density" in sw.columns and "Speed" in sw.columns:
        sw["Flow_Pressure"] = 1.6726e-6 * sw["Density"] * (sw["Speed"] ** 2)
    else:
        sw["Flow_Pressure"] = np.nan

    sw = sw.resample("H").mean(numeric_only=True)
    for c in ["Bz_GSM","Density","Speed","Flow_Pressure"]:
        if c not in sw.columns: sw[c] = np.nan
    return sw[["Bz_GSM","Density","Speed","Flow_Pressure"]]

def fetch_cme_counts(start_utc: datetime, end_utc: datetime) -> int:
    try:
        url = (
            "https://api.nasa.gov/DONKI/CME"
            f"?startDate={start_utc.date()}&endDate={end_utc.date()}&api_key={NASA_KEY}"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return 0
        data = r.json()
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0

# ===== Dst: SWPC Kyoto JSON + OMNI fallback =====
def fetch_dst_swpc(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        url = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
        raw = _safe_swpc_json(url, retries=3, timeout=20)
        if not raw or len(raw) < 2:
            return pd.DataFrame(columns=["Dst"])
        cols = raw[0]
        df = pd.DataFrame(raw[1:], columns=cols)
        tcol = "time_tag" if "time_tag" in df.columns else next((c for c in df.columns if c.lower()=="time_tag"), None)
        dcol = "dst"      if "dst"      in df.columns else next((c for c in df.columns if c.lower()=="dst"), None)
        if not tcol or not dcol: return pd.DataFrame(columns=["Dst"])
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df[dcol] = pd.to_numeric(df[dcol], errors="coerce")
        df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
        df = df.resample("H").mean(numeric_only=True).rename(columns={dcol:"Dst"})
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        return df.loc[mask].copy()
    except Exception:
        return pd.DataFrame(columns=["Dst"])

def fetch_dst_omni(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # JSON
    try:
        url_json = f"https://omniweb.gsfc.nasa.gov/api/hro?start={start_iso}&stop={end_iso}&parameters=Dst"
        r = requests.get(url_json, timeout=30)
        if r.status_code == 200:
            js = r.json()
            if isinstance(js, dict) and "result" in js: js = js["result"]
            times, vals = [], []
            for row in js:
                tt = row.get("Time") or row.get("time_tag") or row.get("EPOCH") or row.get("TimeTag")
                dv = row.get("Dst")  or row.get("dst")      or row.get("DST")
                if tt is None or dv is None: continue
                t = pd.to_datetime(tt, utc=True, errors="coerce")
                if pd.isna(t): continue
                try: v = float(dv)
                except: v = np.nan
                times.append(t); vals.append(v)
            if times:
                df = pd.DataFrame({"Dst": vals}, index=pd.to_datetime(times, utc=True))
                df.index = df.index.tz_convert("UTC").tz_localize(None)
                df = df.sort_index()
                return df
    except Exception:
        pass
    # CSV
    try:
        url_csv = f"https://omniweb.gsfc.nasa.gov/api/hro.csv?start={start_iso}&stop={end_iso}&parameters=Dst"
        r = requests.get(url_csv, timeout=30)
        if r.status_code == 200 and r.text:
            buf = io.StringIO(r.text)
            df = pd.read_csv(buf)
            time_col = next((c for c in ["Time","time_tag","EPOCH","TimeTag","Datetime","Date_UTC"] if c in df.columns), None)
            dst_col  = next((c for c in ["Dst","dst","DST"] if c in df.columns), None)
            if time_col and dst_col:
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
                df = df.dropna(subset=[time_col])
                out = df[[time_col, dst_col]].rename(columns={time_col:"datetime", dst_col:"Dst"})
                out["datetime"] = out["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
                out = out.set_index("datetime").sort_index()
                return out
    except Exception:
        pass
    return pd.DataFrame(columns=["Dst"])

def fetch_dst_with_fallback(start_dt: datetime, end_dt: datetime):
    df = fetch_dst_swpc(start_dt, end_dt)
    if not df.empty: return df, "kyoto"
    df = fetch_dst_omni(start_dt, end_dt)
    if not df.empty: return df, "omni"
    return pd.DataFrame(columns=["Dst"]), "none"

# ========================= Dataset & features =========================
def build_dataset(start_utc: datetime, end_utc: datetime, dst_for_features: pd.DataFrame) -> pd.DataFrame:
    idx  = pd.date_range(start_utc, end_utc, freq="H")
    base = pd.DataFrame(index=idx)
    kp   = fetch_kp_hourly()
    sw   = fetch_solar_wind_hourly()
    df   = base.join(kp, how="left").join(sw, how="left")
    if not dst_for_features.empty:
        df = df.join(dst_for_features, how="left")  # for Dst lags in features

    # placeholders to match your training set
    df["Sunspot_No"] = 50.0
    df["flare_strength_max"] = 0.0
    df["cme_speed_mean"] = 0.0
    df["cme_halfAngle_mean"] = 0.0
    df["Ap"] = df["Kp"].fillna(2.0) * 3.0

    df = normalize_cols(df).ffill().bfill()
    # sane fallbacks
    for c, val in [("Dst", 0.0), ("Bz_GSM", 0.0), ("Density", 5.0), ("Speed", 400.0)]:
        if c not in df.columns or df[c].isna().all(): df[c] = val
    if "Flow_Pressure" not in df.columns or df["Flow_Pressure"].isna().all():
        df["Flow_Pressure"] = 1.6726e-6 * df["Density"] * (df["Speed"] ** 2)
    return df

def make_features_121(df: pd.DataFrame) -> pd.DataFrame:
    """
    EXACT 121-feature schema used by your scaler/model:
    11 base cols × (lags: 1,3,6,12,24 = 5; rolls mean+max over 3,6,12 = 6) → 11×11 = 121
    """
    base_cols = [
        "Dst","Kp","Sunspot_No","cme_speed_mean","flare_strength_max",
        "Bz_GSM","Density","Speed","Flow_Pressure","Ap","cme_halfAngle_mean"
    ]
    for c in base_cols:
        if c not in df.columns: df[c] = 0.0

    feat = {}
    lags  = [1,3,6,12,24]
    rolls = [3,6,12]
    for c in base_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        for lag in lags:
            feat[f"{c}_lag_{lag}hr"] = s.shift(lag)
        for w in rolls:
            feat[f"{c}_roll_mean_{w}hr"] = s.rolling(w).mean()
            feat[f"{c}_roll_max_{w}hr"]  = s.rolling(w).max()
    X = pd.DataFrame(feat, index=df.index).dropna()
    # guard shape for sanity (11 * (5+3+3) = 121)
    assert X.shape[1] == 121, f"Feature shape {X.shape[1]} != 121"
    return X

# ========================= Risk & plotting =========================
def compute_risk(dst):
    if dst < -50: return "RED", "HIGH (Geomagnetic Storm Likely)"
    elif dst < -30: return "YELLOW", "MODERATE"
    return "GREEN", "LOW"

def risk_score_from_dst(dst):
    return float(np.clip(np.interp(-dst, [0, 30, 50, 100], [0, 40, 70, 100]), 0, 100))

def gauge_indicator(value, title, minv, maxv, steps=None, threshold=None, suffix=""):
    g = go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix}, title={"text": title},
        gauge={"axis": {"range": [minv, maxv]}, "bar": {"color": "white"},
               "steps": steps or [], "threshold": threshold or {}}
    )
    fig = go.Figure(g)
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="#0e1117", font=dict(color="#e8eef4"))
    return fig

def plot_3d_sun_with_cmes(cme_count: int):
    u = np.linspace(0, 2*np.pi, 50); v = np.linspace(0, np.pi, 25)
    x = 1.0*np.outer(np.cos(u), np.sin(v)); y = 1.0*np.outer(np.sin(u), np.sin(v))
    z = 1.0*np.outer(np.ones_like(u), np.cos(v))
    sun = go.Surface(x=x, y=y, z=z, colorscale=[[0.0, "#ffcc33"], [1.0, "#ff6600"]],
                     showscale=False, opacity=0.95)
    cones = []
    n = int(min(6, max(0, cme_count)))
    if n > 0:
        thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
        phis   = np.linspace(np.pi/6, np.pi/3, n)
        for i in range(n):
            r0 = 1.1
            ux = np.sin(phis[i]) * np.cos(thetas[i])
            uy = np.sin(phis[i]) * np.sin(thetas[i])
            uz = np.cos(phis[i])
            cones.append(go.Cone(x=[r0*ux], y=[r0*uy], z=[r0*uz],
                                 u=[ux*0.8], v=[uy*0.8], w=[uz*0.8],
                                 colorscale="OrRd", sizemode="absolute", sizeref=0.4,
                                 showscale=False, anchor="tail", opacity=0.9))
    layout = go.Layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                  zaxis=dict(visible=False), aspectmode="data", bgcolor="#0e1117"),
                       paper_bgcolor="#0e1117", margin=dict(l=0, r=0, b=0, t=30),
                       title=f"☀️ Sun & CME Directions (showing {n} of {cme_count})")
    return go.Figure(data=[sun] + cones, layout=layout)

# ========================= Sidebar =========================
with st.sidebar:
    st.header("Date Range (UTC)")
    start_date = st.date_input("Start date", datetime.utcnow() - timedelta(days=4))
    end_date   = st.date_input("End date",   datetime.utcnow())
    run = st.button("Run Forecast")

# ========================= Main =========================
if run:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.min.time()) + timedelta(hours=23)

    # Truth for lags AND backtest
    truth_full, truth_src = fetch_dst_with_fallback(start_dt, end_dt + timedelta(hours=HORIZON_H+2))

    with st.spinner("Fetching live data & preparing dataset..."):
        raw = build_dataset(start_dt, end_dt, truth_full)
        cme_count = fetch_cme_counts(start_dt, end_dt)

    with st.spinner("Engineering features..."):
        feat = make_features_121(raw.copy())
        if feat.empty:
            st.error("Not enough data to build features. Select a wider window (≥ 30 hours).")
            st.stop()

    # Shift truth by +3h to form evaluation target
    target = truth_full["Dst"].shift(-HORIZON_H).reindex(feat.index)

    # Load saved model & scaler (NO refit)
    try:
        model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to load model/scaler: {e}"); st.stop()

    # Align to expected columns
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
    elif hasattr(scaler, "get_feature_names_out"):
        try: expected = list(scaler.get_feature_names_out())
        except Exception: expected = list(feat.columns)
    else:
        try: expected = list(model.get_booster().feature_names or [])
        except Exception: expected = []
        if not expected: expected = list(feat.columns)

    X = feat.reindex(columns=expected, fill_value=0.0)
    try: Xs = scaler.transform(X)
    except Exception: Xs = X.values

    preds = model.predict(Xs)

    # Real backtest metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    mask = ~np.isnan(target.values)
    if mask.sum() == 0:
        st.error("No overlapping target values after shift; try a different window.")
        st.stop()
    r2  = r2_score(target.values[mask], preds[mask])
    mae = mean_absolute_error(target.values[mask], preds[mask])

    # Latest forecast
    pred_series = pd.Series(preds[mask], index=feat.index[mask]).sort_index()
    dst_latest = float(pred_series.iloc[-1])
    cond, risk_text = compute_risk(dst_latest)
    risk_score = risk_score_from_dst(dst_latest)

    # Gauges
    cA, cB, cC = st.columns(3)
    with cA:
        st.subheader("Dst Gauge")
        steps = [{"range": [-150,-50], "color":"#4a1c1c"},
                 {"range": [-50,-30], "color":"#463c10"},
                 {"range": [-30,  0], "color":"#103d1f"}]
        fig_dst = gauge_indicator(dst_latest, f"Dst (nT) — +{HORIZON_H}h", -150, 0,
                                  steps=steps, threshold={"line":{"color":"white","width":3},"value":dst_latest})
        st.plotly_chart(fig_dst, use_container_width=True)
    with cB:
        st.subheader("Kp Gauge")
        last_kp = float(raw["Kp"].iloc[-1]) if "Kp" in raw.columns and not raw["Kp"].tail(1).isna().all() else 0.0
        steps_kp = [{"range":[0,4], "color":"#103d1f"},
                    {"range":[4,6], "color":"#463c10"},
                    {"range":[6,9], "color":"#4a1c1c"}]
        fig_kp = gauge_indicator(last_kp, "Kp (now)", 0, 9,
                                 steps=steps_kp, threshold={"line":{"color":"white","width":3},"value":last_kp})
        st.plotly_chart(fig_kp, use_container_width=True)
    with cC:
        st.subheader("Storm Risk Gauge")
        steps_risk = [{"range":[0,40], "color":"#103d1f"},
                      {"range":[40,70], "color":"#463c10"},
                      {"range":[70,100],"color":"#4a1c1c"}]
        fig_risk = gauge_indicator(risk_score, f"Risk ({cond})", 0, 100,
                                   steps=steps_risk, threshold={"line":{"color":"white","width":3},"value":risk_score}, suffix="%")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Snapshot
    st.subheader("Latest Live Snapshot")
    last = raw.tail(1).iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Kp", f"{last.get('Kp', np.nan):.2f}")
    m2.metric("Bz (GSM) nT", f"{last.get('Bz_GSM', np.nan):.2f}")
    m3.metric("Speed (km/s)", f"{last.get('Speed', np.nan):.1f}")
    m4.metric("Density (p/cc)", f"{last.get('Density', np.nan):.2f}")
    m5.metric("Flow Pressure (nPa)", f"{last.get('Flow_Pressure', np.nan):.2f}")

    # Accuracy
    st.markdown("---")
    st.subheader("📊 Backtest (this window, SWPC Kyoto → OMNI fallback)")
    src_label = {"kyoto":"SWPC kyoto-dst.json","omni":"OMNI (NASA SPDF)","none":"Unknown"}.get(truth_src, "Unknown")
    st.caption(f"Truth source: {src_label} | Horizon: +{HORIZON_H} h | Overlap: {mask.sum()} points")
    c1, c2 = st.columns(2)
    c1.metric("R²",  f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f} nT")

    comp = pd.DataFrame({
        f"Actual Dst (+{HORIZON_H}h)": target.values,
        f"Predicted Dst (+{HORIZON_H}h)": preds
    }, index=feat.index).dropna()
    if not comp.empty:
        st.line_chart(comp)

    # Time series
    st.markdown("---")
    st.subheader("Time Series (selected window)")
    ts_cols = [c for c in ["Kp","Bz_GSM","Speed","Density","Flow_Pressure","Dst"] if c in raw.columns]
    if ts_cols:
        st.line_chart(raw[ts_cols])

    # 3D Sun & CMEs
    st.markdown("---")
    st.subheader("3D Sun & CME Visualization")
    st.plotly_chart(plot_3d_sun_with_cmes(cme_count), use_container_width=True)

    st.markdown("---")
    st.caption("Sources: NOAA SWPC (Kp, Solar Wind, Kyoto Dst JSON) • NASA DONKI (CMEs). Verify critical decisions with official alerts.")
else:
    st.info(f"Pick a UTC date range and click **Run Forecast**. Horizon is fixed at **+{HORIZON_H} h** (matches your model).")
