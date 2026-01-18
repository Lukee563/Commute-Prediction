import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_xgboost(data):
    X = data.drop(columns=['target', 'otime'])
    y = data['target']

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    # Train Test split:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123
    )

    model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,          # Increase estimators...
    learning_rate=0.1,         
    max_depth=6,                # Slightly shallower trees prevent overfitting noisy survey data
    
    # --- Regularization 
    reg_lambda=1.5,             # L2 regularization (Ridge-like)
    reg_alpha=0.5,              # L1 regularization (Lasso-like)
    
    subsample=0.7,              # Row sampling
    colsample_bytree=0.7,       # Column sampling per tree
    colsample_bylevel=0.7,      # Column sampling per level
    
    tree_method='hist',
    random_state=123,
    n_jobs=-1                   # Use all CPU cores
)

    # Fit (no eval params)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)

    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    return model, results