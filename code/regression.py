from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

def train_linear_regression(data):
    X = data.drop(columns=['target', 'otime']) 
    y = data['target']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    # Train Test split:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123
    )
    
    model = LinearRegression().fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Final Test Mean Squared Error: {mse:.4f}")

    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    
    return model, results