import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data.CSV")
plt.figure(figsize=(8, 5))

plt.scatter(
    df["hours_studied"],
    df["attendance"],
    c=df["pass"],
    s=100,
    edgecolors="black"
)

plt.xlabel("Hours Studied")
plt.ylabel("Attendance")
plt.title("Student Success Prediction - Pass/Fail Visualization")

plt.colorbar(label="Pass Status: 0 = Fail, 1 = Pass")

plt.show()

X = df[
    [
        "hours_studied",
        "attendance",
        "assignments_completed"
    ]
]

y = df["pass"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

hours = int(input("Enter the Hours : "))
attendence = float(input("Enter the attendence : "))
assign = int(input("Enter how many assingments you completed : "))

new_student = pd.DataFrame(
    {
        "hours_studied": [hours],
        "attendance": [attendence],
        "assignments_completed": [assign]
    }
)

result = model.predict(new_student)

if result[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")