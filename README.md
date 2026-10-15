# 📈 ML Zoomcamp 2024: Time-Series Sales Forecasting

An end-to-end Machine Learning pipeline developed for the ML Zoomcamp 2024. This project evaluates daily sales forecasting performance by comparing a naive seasonal baseline against a Random Forest Regressor using time-based validation.

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

## 📚 Concepts & Modeling Methodology

### 1. Feature Engineering: Lag Variables
To convert sequential time-series data into a supervised machine learning format, we generated **lag features**:
- **`lag_1`:** Sales from 1 day prior (captures short-term daily continuity).
- **`lag_7`:** Sales from 7 days prior (captures 7-day weekly seasonality).
- **`rolling_mean_7` & `rolling_mean_28`:** Moving averages over 1 and 4-week windows to capture overall trend shifts.

### 2. Baseline Benchmark: Naive Lag-7 Predictor
Before deploying machine learning, we established a heuristic baseline: $\hat{y}_t = y_{t-7}$. This assumes sales today will equal sales on the same weekday last week. In retail environments with strong weekly seasonality, this simple rule provides a strict benchmark for evaluating complex models.

### 3. Model Architecture: Random Forest Regressor
We trained a `RandomForestRegressor` ensemble consisting of 100 decision trees. The model splits on non-linear feature interactions (such as combining calendar flags with moving averages) to predict daily demand without requiring normalized input distributions.

### 4. Validation Strategy: Time-Series Split
Unlike standard randomized cross-validation—which causes **data leakage** by leaking future information into past training samples—we used a strict **temporal split**:
- **Training Period:** First 703 days (Historical learning baseline)
- **Validation Period:** Final 30 days (Simulating real-world multi-step future forecasting)

---


## 🏆 Model Performance & Evaluation

Models were evaluated over the final 30-day holdout set (2024-08-28 to 2024-09-26):

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | Key Trait |
| :--- | :---: | :---: | :--- |
| **Naive Seasonal Baseline (Lag-7)** | **3,607.34** | **4,929.73** | Directly captures 7-day cyclical seasonality |
| **Random Forest Regressor** | 4,028.41 | 5,277.07 | Over-relies on `lag_1`; subject to recursive drift |

<img width="1248" height="558" alt="image" src="https://github.com/user-attachments/assets/7194049d-a63c-4653-b436-a256be7a1f4a" />

**Why Did the Simple Baseline Win?**
- **Strong Weekly Seasonality**: The sales curve shows massive 7-day cyclical peaks (e.g., weekend spikes around 2024-09-05, 2024-09-13, and 2024-09-20). Simply copying last week's exact value captures this pattern effortlessly.
- **Recursive Feature Drift**: Multi-step forecasting into the future causes lag features (`lag_1`) to rely on previous predictions rather than ground truth, leading error to compound over 30 days.
- **Mid-Month Anomaly Spikes**: Notice the spike on 2024-09-16 in the Random Forest forecast that didn't happen in real life. The model over-predicted based on historical calendar features without promotion or holiday context.

---

## 🔍 Feature Importance Diagnostics

Feature importance analysis of the trained Random Forest model highlighted:
1. **`lag_1` (59.8%):** Dominant predictor, leading to recursive error accumulation over multi-step horizons.
2. **`rolling_mean_28` (11.6%) & `rolling_mean_7` (11.6%):** Captured long-term moving baseline trends effectively.
3. **`lag_7` (3.3%):** Under-weighted relative to its actual predictive power in strongly seasonal series.

---

## 💡 Key Lessons

1. **Baselines are Benchmark Drivers:** Complex models must prove superiority over simple domain baselines before deployment.
2. **Multi-step Forecasting Drift:** Recursive autoregressive models degrade over long horizons. Direct multi-step models or LightGBM tuned with explicit weekly interaction terms are recommended next steps.
3. **Exogenous Integration:** Future iterations will merge store metadata (`stores.csv`) and promotional history (`discounts_history.csv`) to resolve unpredicted mid-month sales spikes.
