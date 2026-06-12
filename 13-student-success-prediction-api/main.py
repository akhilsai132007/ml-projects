import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Student Success Prediction API",
    description="A FastAPI app that predicts whether a student will PASS or FAIL.",
    version="1.0"
)

model = joblib.load("student_success_model.pkl")

class StudentData(BaseModel):
    hours_studied: float
    attendance: float
    assignments_completed: int

@app.get("/")
def home():
    return {
        "message": "Student Success Prediction API is running"
    }

@app.post("/predict")
def predict_student(data: StudentData):
    input_data = pd.DataFrame({
        "hours_studied": [data.hours_studied],
        "attendance": [data.attendance],
        "assignments_completed": [data.assignments_completed]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return {
        "prediction": result
    }