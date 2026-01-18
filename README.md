# Predicting the Commutes of 1990 San Francisco Bay Area Drivers
### Supervised Learning: Parametric Regularization vs. Non-Parametric Ensemble Methods

Author: Luke Catalano

Affiliation: University of California, Santa Cruz, M.S. Quantitative Economics & Finance

## Research Overview
This project applies machine learning techniques to predict individual commute durations using the **1990 Bay Area Travel Survey**. The core objective is to compare the predictive accuracy of parametric models constrained by linearity (Lasso and Ridge) against non-parametric models (Random Forest) to determine the extent of non-linear interactions in urban transportation data.

## Economic & Practical Motivation
* **Infrastructure Planning:** Accurate trip-time estimates are vital for civil engineers and urban planners to manage highway capacity and identify geographic bottlenecks.
* **Labor Market Dynamics:** Commute times act as a "tax" on labor; understanding these determinants helps model worker decisions regarding residential location versus employment centers.
* **Algorithmic Benchmarking:** Testing whether modern non-parametric methods significantly outperform traditional econometric specifications in high-dimensional spatial data.

## Methodology

### 1. Parametric Regularized Regression
To address high-dimensional features and potential multicollinearity (especially between geographic indicators), I implemented **Lasso (L1)** and **Ridge (L2)** regressions. These models introduce a penalty term to the OLS objective function:

$$\text{Lasso: } \min_{\beta} \sum_{i=1}^{n} (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$
$$\text{Ridge: } \min_{\beta} \sum_{i=1}^{n} (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

* **Feature Selection:** Lasso successfully zeroed out less relevant features, isolating travel mode and county-pair interactions as the primary predictors.
* **Results:** Both models achieved an Out-of-Sample (OOS) $R^2$ of approximately **0.15**, performing moderately better than a null baseline but limited by the linear functional form.

* **Final Feature Set:** After writing my initial paper, I have heavily modified the feature set to simulate actual work commutes, primarily be setting strict guidelines as to what can be considered a moderate commute (20 - 150 minutes), reducing noise and simplifying analysis. 
### 2. Non-Parametric Random Forest
I deployed a **Random Forest** ensemble consisting of 500 trees with a minimum node size of 10. Unlike the linear models, this approach does not assume a specific functional form for the relationship between variables like income, start time, or geographic origin.

* **Performance:** The Random Forest achieved an OOS $R^2$ of **0.2343**.
* **Interpretation:** The 4 percentage point improvement over parametric models suggests that the true data-generating process contains non-linearities (e.g., peak-hour congestion effects) that linear models fail to capture.

### 3. XGBoost, a Gradient Boosting Ensemble 
After writing my analysis, I decided to implement XGboost as a final attempt at maximizing predicitive performance after poor predictive capability iwth Lasso/Ridge and Random Forest models. 

* **Performance:** The XGBoost achieved an OOS $R^2$ of **0.646**.
* **Interpretation:** This suggests that the commute duration in the 1990 Bay Area data is not just non-linear, but involves highly complex additive interactions that Gradient Boosting is uniquely suited to optimize via gradient descent in the functional space.

## Data & Feature Engineering
* **Source:** 1990 Bay Area Travel Survey.
* **Preprocessing:** Handled categorical variables for travel mode (car, motorcycle, carpool) and home/work county locations.
* **Interactions:** In the parametric models, I engineered interaction terms for work/home county pairs to capture spatial friction across the Bay Area.
* **Outlier Management:** Log-transformed commute durations were evaluated to stabilize variance, though raw duration was the final target for interpretability.

## Key Findings
* **The "Carpool" Surprise:** Contrary to the hypothesis that coordinating pickups would significantly increase trip time, the models found carpooling was only associated with a **~3.5 minute increase**, holding all else constant. 
* **Spatial Friction:** The home-work county interactions were the strongest predictors, highlighting the heavy influence of regional geography over individual demographic characteristics.
* **Bias-Variance Tradeoff:** The superior performance of the ensemble method highlights a clear tradeoff: while Lasso/Ridge offered high interpretability of coefficients, Random Forest offered significantly higher predictive utility by capturing latent interactions.

## Tech Stack
* **Python:** Core programming environment.
* **Scikit-Learn:** Used for Lasso, Ridge, and Random Forest implementations.
* **Pandas/NumPy:** Data manipulation and feature engineering.
