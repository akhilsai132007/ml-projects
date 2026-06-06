import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.float_format", "{:.2f}".format)

df = pd.read_csv("house_data.csv")

print("Dataset Preview:")
print(df.head())

X = df[["house_size", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

results["Prediction Error"] = (
    results["Actual Price"] - results["Predicted Price"]
).abs()

print("\nActual vs Predicted Prices:")
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R2 Score: {r2:.4f}")

print("\nModel Details:")
print("House Size Coefficient:", round(model.coef_[0], 2))
print("Bedrooms Coefficient:", round(model.coef_[1], 2))
print("Intercept:", round(model.intercept_, 2))

new_house = pd.DataFrame({
    "house_size": [2400],
    "bedrooms": [2]
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print(new_house)

print("\nPredicted Price for New House:")
print(f"₹{predicted_price[0]:,.2f}")