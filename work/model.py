import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_xgboost(data):
    X = data.drop(columns=['target', 'dtime'])
    y = data['target']

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    # Train Test split:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123
    )

    #
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        seed=123,
        n_estimators=1000,
        eval_metric='rmse',                                                                           
    )

    # Fit (no eval params here anymore)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Final Test Mean Squared Error: {mse:.4f}")

    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    return model, results