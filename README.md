# 🌪️ Kuwait Dust Storm Early Warning System (PM10 Predictive Radar)

A production-grade machine learning web application built to forecast severe particulate matter ($\text{PM}_{10}$) spikes and dust storms in Kuwait. Designed using meteorological physics and time-series feature engineering, this tool serves as an automated early-warning radar for environmental monitoring.

---

## 🚀 Project Overview
Kuwait's hyper-arid climate and vulnerability to severe dust storms make real-time air quality forecasting critical for public health. This project moves beyond reactive air quality sensors by building a predictive regression model that analyzes meteorological drivers (wind speed, humidity, temperature fronts) to anticipate extreme dust events hours in advance.

---

## 🛠️ Key Technical Features
* **Time-Series Feature Engineering**: Extracted rolling statistics (6-hour and 12-hour windows) and multi-step lag features to capture the temporal autocorrelation of atmospheric pressure and wind patterns.
* **Extreme Event Tuning (Custom Sample Weights)**: Standard regression models often under-predict rare, hazardous spikes. Applied custom sample weight penalties to hazardous $\text{PM}_{10}$ thresholds ($>300 \, \mu\text{g/m}^3$) to force the algorithm to prioritize high-risk storm detection.
* **Gradient Boosting Architecture**: Trained using an optimized **XGBoost Regressor** with early-stopping mechanics to prevent overfitting on historical climate data.

- --

## 📂 Project Structure
```text
Dust-Storm-App/
│
├── scripts                    # scripts to use before training
├── data     # data used to train models
├── model           # scripts for running final model
└── README.md                  # Project documentation
