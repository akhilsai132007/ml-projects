import pandas as pd

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")

X = df[["hours_studied","attendance","assignments_completed"]]
y = df["pass"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2
)

pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model" , LogisticRegression())
])

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5
)

print("\n===== CROSS VALIDATION SCORES =====")
print(cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n===== TEST RESULTS =====")
print("Actual Values:", y_test.values)
print("Predictions:", predictions)
print("Test Accuracy:", accuracy)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions))

new_student = pd.DataFrame({
    "hours_studied": [6],
    "attendance": [78],
    "assignments_completed": [7]
})

new_prediction = pipeline.predict(new_student)

print("\n===== NEW STUDENT PREDICTION =====")

if new_prediction[0] == 1:
    print("Prediction: Student will PASS")
else:
    print("Prediction: Student will FAIL")

