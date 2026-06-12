import pandas as pd
import joblib

model = joblib.load("student_success_model.pkl")

hours_studied = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance percentage: "))
assignments_completed = int(input("Enter assignments completed: "))

new_student = pd.DataFrame({
    "hours_studied": [hours_studied],
    "attendance": [attendance],
    "assignments_completed": [assignments_completed]
})

prediction = model.predict(new_student)

print("\n===== PREDICTION RESULT =====")

if prediction[0] == 1:
    print("Student is likely to PASS")
else:
    print("Student is likely to FAIL")
    