# 🎓 Student Success Prediction API 🚀

A Machine Learning API built using FastAPI that predicts whether a student is likely to PASS or FAIL based on study habits and academic performance.

This project combines Machine Learning, Model Persistence, and FastAPI to simulate how ML models are used in real-world applications.

---

# 📌 Features

✅ Train Machine Learning Model

✅ Save Trained Model (.pkl)

✅ Load Saved Model

✅ FastAPI Integration

✅ User Input Validation

✅ Pass/Fail Prediction

✅ JSON Response

✅ Interactive API Documentation

---

# 📊 Dataset Features

The model uses the following features:

| Feature               | Description                     |
| --------------------- | ------------------------------- |
| hours_studied         | Number of study hours           |
| attendance            | Attendance percentage           |
| assignments_completed | Number of assignments completed |

Target:

| Value | Meaning |
| ----- | ------- |
| 1     | PASS    |
| 0     | FAIL    |

---

# 🧠 Machine Learning Workflow

```text
Dataset
↓
Train Model
↓
Save Model (.pkl)
↓
FastAPI
↓
User Input
↓
Prediction
↓
JSON Response
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Random Forest Classifier
* StandardScaler
* Pipeline
* Joblib
* FastAPI
* Uvicorn

---

# 📂 Project Structure

```text
13-student-success-prediction-api/

│
├── data.csv
├── train_model.py
├── main.py
├── student_success_model.pkl
├── requirements.txt
└── README.md
```

---

# 💾 Model Persistence

The trained model is saved using:

```python
joblib.dump(
    pipeline,
    "student_success_model.pkl"
)
```

The model is loaded inside FastAPI using:

```python
model = joblib.load(
    "student_success_model.pkl"
)
```

This allows predictions without retraining.

---

# 🚀 How To Run

## Step 1: Train and Save Model

```bash
python train_model.py
```

This creates:

```text
student_success_model.pkl
```

---

## Step 2: Start FastAPI Server

```bash
uvicorn main:app --reload
```

---

## Step 3: Open API Docs

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically generates Swagger UI documentation.

---

# 📥 Example Request

```json
{
  "hours_studied": 8,
  "attendance": 90,
  "assignments_completed": 12
}
```

---

# 📤 Example Response

```json
{
  "prediction": "PASS"
}
```

---

# 🎯 Learning Outcomes

Through this project, I learned:

* FastAPI Fundamentals
* Machine Learning APIs
* Model Persistence
* Joblib
* Serialization
* Deserialization
* API Endpoints
* Request Validation
* JSON Responses
* Production-Style ML Workflow

---

# 🌍 Real-World Application Architecture

```text
User
↓
FastAPI
↓
Load Saved Model
↓
Prediction
↓
JSON Response
```

This is the same workflow used in recommendation systems, fraud detection systems, customer analytics platforms, and many other Machine Learning applications.

---

# 🏴‍☠️ Project Goal

Build a real Machine Learning API capable of:

```text
Train Once
↓
Save Model
↓
Load Model
↓
Predict Forever
```

This project bridges the gap between Machine Learning and Backend Development.
