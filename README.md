# Predicting Bay Area Commute Durations
**Supervised Learning | Parametric vs. Non-Parametric Methods | Python**

---

## Overview

This project applies supervised machine learning to predict individual commute durations using the **1990 Bay Area Travel Survey**. The core question is whether non-parametric ensemble methods capture meaningful non-linearities in urban transportation data that regularized linear models cannot, and by how much.

Three model classes are compared in sequence: **Lasso (L1)** and **Ridge (L2)** regularized regression, a **Random Forest** ensemble, and a **gradient-boosted XGBoost** model. The progression tests the bias-variance tradeoff in a high-dimensional spatial setting, and the performance gap between approaches turns out to be substantial.

---

## Key Results

| Model | OOS R² | Notes |
|---|---|---|
| Lasso (L1) | 0.15 | 20-fold CV, one-hot county pairs |
| Ridge (L2) | 0.15 | 20-fold CV |
| Random Forest | 0.23 | 1,000 trees, min leaf = 10 |
| **XGBoost** | **0.646** | Best model - see below |

**XGBoost out-of-sample performance on held-out test set (n = 3,035):**
- Accuracy (1 − MAPE): **64.6%**
- Mean absolute error: **16.6 minutes**
- Median absolute error: **11.9 minutes**

The 40+ percentage point gap between linear and gradient-boosted models indicates the true data-generating process contains highly complex additive interactions, particularly peak-hour congestion effects and county-pair spatial friction, that linear functional forms cannot recover.

---

## Methodology

**Target variable:** Log-transformed commute duration (minutes), restricted to work commutes between 20 and 150 minutes to remove noise from non-standard trips.

**Feature engineering:**
- Departure time encoded as cyclical features (`hour_sin`, `hour_cos`) to capture peak-hour effects without discontinuity at midnight
- Home × work county interactions as route identifiers
- Categorical encoding for travel mode, occupation, vehicle type, business type, and trip number

**Model details:**
- Lasso/Ridge: 20-fold cross-validation for λ selection; StandardScaler applied; one-hot encoding with county-pair interaction terms
- Random Forest: 1,000 trees, `min_samples_leaf=10`, parallelized
- XGBoost: 1,000 estimators, L1 + L2 regularization (`alpha=0.5`, `lambda=1.5`), row/column subsampling at 0.7, histogram-based tree method

**Key findings:**
- Spatial friction (home-work county pairs) was the strongest predictor class across all models
- Carpooling was associated with only ~3.5 additional minutes, contrary to the prior expectation that pickup coordination would meaningfully extend trip time
- The Random Forest's modest improvement over linear models (+8 pp) versus XGBoost's dramatic improvement (+43 pp) suggests gradient boosting's sequential error-correction is particularly well-suited to the additive structure of commute data

---

## Repository Structure

```
├── code/
│   ├── data/
│   │   └── data.tsv              # 1990 Bay Area Travel Survey (raw)
│   ├── data_prep.py              # Cleaning, feature engineering, log transform
│   ├── main.py                   # Entry point, runs XGBoost pipeline end-to-end
│   ├── XGBoost.py                # XGBoost model definition and training
│   ├── models.py                 # Lasso, Ridge, and Random Forest comparison
│   ├── regression.py             # Baseline linear regression
│   └── eda.py                    # Exploratory data analysis and scatter plots
├── results_interpreted.csv  # XGBoost predictions on test set (actual vs. predicted minutes)
├── Commute_Prediction_Paper.pdf  # Final empirical paper
└── README.md
```

**To run:**
```bash
pip install pandas numpy scikit-learn xgboost
cd code
python main.py        # Runs XGBoost, prints accuracy, saves results_interpreted.csv
python models.py      # Runs Lasso, Ridge, and Random Forest comparison
```

---

## Data

**Source:** 1990 Bay Area Travel Survey, published by the Metropolitan Transportation Commission. The dataset contains individual-level trip records including origin/destination times, travel mode, county of residence and employment, vehicle type, and demographic characteristics.

**Preprocessing decisions:**
- Restricted to work-purpose destination trips (`dpurp == 1`)
- Commute window: 20–150 minutes (removes extreme outliers and non-standard commutes)
- Age capped at 100 to remove data entry errors
- Log transformation applied to commute duration to stabilize variance

---

## Tech Stack

Python, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`
