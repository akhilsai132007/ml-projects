import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("student_pass_data.csv")

print("\n===== DATASET PREVIEW =====")
print(df.head())

X = df[["hours_studied", "attendance"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("\n===== TEST RESULTS =====")

results = pd.DataFrame({
    "Actual Result": y_test.values,
    "Predicted Result": predictions
})

print(results)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {round(accuracy * 100, 2)} %")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\n===== STUDENT PASS PREDICTION =====")

hours = float(input("Enter Hours Studied: "))
attendance = float(input("Enter Attendance Percentage: "))

new_student = pd.DataFrame({
    "hours_studied": [hours],
    "attendance": [attendance]
})

result = model.predict(new_student)

print("\n===== FINAL PREDICTION =====")

if result[0] == "Yes":
    print("Prediction: PASS ✅")
else:
    print("Prediction: FAIL ❌")

df["pass_numeric"] = df["pass"].map({
    "No": 0,
    "Yes": 1
})

plt.figure(figsize=(8, 5))

plt.scatter(
    df["hours_studied"],
    df["pass_numeric"],
    label="Student Data"
)

plt.title("Hours Studied vs Pass/Fail")
plt.xlabel("Hours Studied")
plt.ylabel("Pass = 1, Fail = 0")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()