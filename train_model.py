import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("dataset/mobile_prices.csv")

print("Dataset:")
print(df.head())


# ==========================================
# 2. Select Features and Target
# ==========================================

X = df[
    [
        "RAM",
        "Storage",
        "Battery",
        "ScreenSize",
        "Camera",
        "ProcessorScore"
    ]
]

y = df["Price"]


# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. Create Linear Regression Model
# ==========================================

model = LinearRegression()


# ==========================================
# 5. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# ==========================================
# 6. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)


# ==========================================
# 7. Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# ==========================================
# 8. Save Model
# ==========================================

joblib.dump(model, "model/linear_regression_model.pkl")

print("\nModel saved successfully!")