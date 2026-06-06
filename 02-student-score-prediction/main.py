import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student_scores.csv")

X = df[["hours_studied"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Actual Scores:")
print(y_test.values)

print("\nPredicted Scores:")
print(pred.round(2))

print("\nModel Evaluation:")
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("MSE:", round(mean_squared_error(y_test, pred), 2))
print("R2 Score:", round(r2_score(y_test, pred), 2))

print("\nModel Details:")
print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

hours = float(input("\nEnter Hours Studied: "))

new_data = pd.DataFrame(
    {
        "hours_studied": [hours]
    }
)

pred_new = model.predict(new_data)

print(f"\nIf a student studies {hours} hours,")
print("Predicted Score:", round(pred_new[0], 2))