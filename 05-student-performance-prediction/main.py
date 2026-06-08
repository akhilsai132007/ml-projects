import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_performance.csv")

X = df[["hours_studied", "attendance", "assignments_completed"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nActual Answers:")
print(y_test.values)

print("\nPredicted Answers:")
print(pred)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)

hours = float(input("\nEnter the Hours Studied: "))
attend = float(input("Enter the Attendance: "))
assign = int(input("Enter how many assignments you completed: "))

new_student = pd.DataFrame({
    "hours_studied": [hours],
    "attendance": [attend],
    "assignments_completed": [assign]
})

new_pred = model.predict(new_student)

if new_pred[0] == 1:
    print("\nPredicted Result: Pass")
else:
    print("\nPredicted Result: Fail")

plt.figure(figsize=(8, 5))
plt.scatter(df["hours_studied"], df["pass"])
plt.title("Hours Studied vs Pass/Fail")
plt.xlabel("Hours Studied")
plt.ylabel("Pass (1 = Pass, 0 = Fail)")
plt.grid(True)
plt.show()