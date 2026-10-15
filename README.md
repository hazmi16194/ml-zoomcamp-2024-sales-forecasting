# ml-zoomcamp-2024-sales-forecasting
Time-series daily sales forecasting pipeline comparing Naive Lag-7 Baseline vs Random Forest Regressor.

# 📈 ML Zoomcamp 2024: Time-Series Sales Forecasting

An end-to-end Machine Learning pipeline developed for the ML Zoomcamp 2024 Competition. This project evaluates daily sales forecasting performance by comparing a naive seasonal baseline against a Random Forest Regressor using time-based validation.

---

## 📌 Executive Summary

- **Objective:** Forecast daily sales demand over a 30-day horizon using historical transactional data (`sales.csv`, ~7.4M rows).
- **Validation Strategy:** Strict 30-day time-series split (Train: 703 days | Validation: 30 days).
- **Key Finding:** The Naive Lag-7 Baseline outperformed the Random Forest Regressor (**MAE: 3,607.34 vs 4,028.41**). Feature importance diagnostics revealed that Random Forest over-indexed on short-term continuity (`lag_1` at ~59.8%) while under-weighting weekly seasonality (`lag_7` at ~3.3%).

---

## 📊 Dataset & Exploratory Data Analysis

### Summary Statistics & Target Distribution
- **Target Variable (`quantity`):** Mean = 55,607 units | Median = 45,420 units | Skewness = 0.88.
- **Distribution Profile:** Bimodal demand curve with distinct weekday vs. weekend operating regimes.
- **Outliers:** Isolated demand spikes observed up to 154,066 units (handled naturally by non-linear tree splits).

### Data Pipeline & Cleaning
1. **Timeline Reindexing:** Verified daily continuity across 733 contiguous days; missing dates reindexed with zero-fill.
2. **Feature Engineering:**
   - **Lags:** `lag_1`, `lag_7`, `lag_14`, `lag_28`
   - **Rolling Window Stats:** 7-day and 28-day moving average (`rolling_mean_7`, `rolling_mean_28`) and standard deviation (`rolling_std_7`).
   - **Calendar Signals:** `day_of_week`, `month`, `quarter`, `is_weekend`.

---

## 🏆 Model Performance & Evaluation

Models were evaluated over the final 30-day holdout set (2024-08-28 to 2024-09-26):

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | Key Trait |
| :--- | :---: | :---: | :--- |
| **Naive Seasonal Baseline (Lag-7)** | **3,607.34** | **4,929.73** | Directly captures 7-day cyclical seasonality |
| **Random Forest Regressor** | 4,028.41 | 5,277.07 | Over-relies on `lag_1`; subject to recursive drift |

---

## 🔍 Feature Importance Diagnostics

Feature importance analysis of the trained Random Forest model highlighted:
1. **`lag_1` (59.8%):** Dominant predictor, leading to recursive error accumulation over multi-step horizons.
2. **`rolling_mean_28` (11.6%) & `rolling_mean_7` (11.6%):** Captured long-term moving baseline trends effectively.
3. **`lag_7` (3.3%):** Under-weighted relative to its actual predictive power in strongly seasonal series.

---

## 💡 Key Lessons & Interview Takeaways

1. **Baselines are Benchmark Drivers:** Complex models must prove superiority over simple domain baselines before deployment.
2. **Multi-step Forecasting Drift:** Recursive autoregressive models degrade over long horizons. Direct multi-step models or LightGBM tuned with explicit weekly interaction terms are recommended next steps.
3. **Exogenous Integration:** Future iterations will merge store metadata (`stores.csv`) and promotional history (`discounts_history.csv`) to resolve unpredicted mid-month sales spikes.
