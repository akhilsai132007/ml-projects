import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("salary_data.csv")

print("\n===== DATASET PREVIEW =====")
print(df.head())

# Features and Target
X = df[["experience"]]
y = df["salary"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on Test Data
predictions = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== TEST RESULTS =====")

results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": predictions.astype(int)
})

print(results)

print("\n===== MODEL EVALUATION =====")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"R² Score : {r2:.4f}")

# User Prediction
print("\n===== SALARY PREDICTION =====")

experience = float(input("Enter Years of Experience: "))

new_data = pd.DataFrame({
    "experience": [experience]
})

predicted_salary = model.predict(new_data)

print(
    f"\nEstimated Salary for {experience} years "
    f"of experience: ₹{int(predicted_salary[0]):,}"
)

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    model.predict(X),
    label="Regression Line"
)

plt.title("Experience vs Salary Prediction")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()