# 🌌 Space Weather Forecasting using Machine Learning

This project predicts geomagnetic storm activity (Dst index) using real-time data from NASA and NOAA.

## 🚀 Features
- Real-time API data (NASA DONKI, NOAA SWPC)
- Predicts Dst index (+3 hours ahead)
- XGBoost ML model
- 121 engineered features
- Interactive Streamlit dashboard

## 📊 Performance
- R²: ~0.62
- MAE: ~7.5 nT

## 🧠 Tech Stack
Python, XGBoost, Pandas, Streamlit, Plotly

## ▶️ Run Locally
```bash
streamlit run app.py