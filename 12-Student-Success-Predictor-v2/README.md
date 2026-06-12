# 🎓 Student Success Predictor v2 🚀

A Machine Learning project that predicts whether a student is likely to pass or fail based on study habits and academic performance.

This version introduces **Model Persistence**, allowing the model to be trained once, saved to disk, loaded later, and reused for predictions without retraining.

---

## 📌 Features

✅ Data Loading with Pandas

✅ Data Visualization using Matplotlib

✅ Train-Test Split

✅ Data Scaling using StandardScaler

✅ Pipeline Workflow

✅ Random Forest Classification

✅ Model Evaluation

✅ Model Persistence using joblib

✅ Save Trained Model (.pkl)

✅ Load Saved Model

✅ User Input Prediction

---

## 📊 Dataset Features

The model uses the following features:

* Hours Studied
* Attendance Percentage
* Assignments Completed

Target:

* Pass (1)
* Fail (0)

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-Learn
* Joblib

---

## 📂 Project Structure

```text
12-student-success-predictor-v2/
│
├── data.csv
├── train_model.py
├── predict.py
├── student_success_model.pkl
├── requirements.txt
├── README.md
└── student_distribution.png
```

---

## 🚀 Workflow

```text
Load Dataset
↓
Visualize Data
↓
Train Model
↓
Create Pipeline
↓
Evaluate Model
↓
Save Pipeline (.pkl)
↓
Load Pipeline
↓
Take User Input
↓
Predict Result
```

---

## 💾 Model Persistence

Instead of retraining the model every time:

```text
Train Once
↓
Save Model
↓
Load Anytime
↓
Predict Forever
```

The trained pipeline is saved using:

```python
joblib.dump(pipeline, "student_success_model.pkl")
```

and loaded using:

```python
pipeline = joblib.load("student_success_model.pkl")
```

---

## ▶️ How to Run

### Train and Save Model

```bash
python train_model.py
```

### Load Model and Predict

```bash
python predict.py
```

---

## 📈 Visualization

The project includes a bar chart showing the distribution of students who passed and failed.

This helps understand class balance before training the model.

---

## 🎯 Learning Outcomes

Through this project, I learned:

* Machine Learning Pipelines
* StandardScaler
* Random Forest Classifier
* Model Evaluation
* Model Persistence
* Serialization and Deserialization
* Saving and Loading Models
* Production-Style Prediction Workflow

---

## 🏴‍☠️ Project Goal

Build a Machine Learning system that can:

```text
Train Once
↓
Save Model
↓
Load Model
↓
Predict Forever
```

This simulates how Machine Learning models are used in real-world applications.
