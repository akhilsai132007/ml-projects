# 🚀 Pipeline Student Success Predictor

## Overview

This Machine Learning project predicts whether a student will pass or fail based on:

* Hours Studied
* Attendance
* Assignments Completed

The project demonstrates a professional Machine Learning workflow using Scikit-Learn Pipelines.

---

## Features

* Data Loading with Pandas
* Train-Test Split
* StandardScaler
* Pipeline
* Logistic Regression
* Cross Validation
* Accuracy Evaluation
* Classification Report
* New Student Prediction

---

## Dataset

Columns:

| Column                | Description                          |
| --------------------- | ------------------------------------ |
| hours_studied         | Number of study hours                |
| attendance            | Attendance percentage                |
| assignments_completed | Number of assignments completed      |
| pass                  | Target variable (1 = Pass, 0 = Fail) |

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn

---

## Machine Learning Workflow

Data
↓
Train-Test Split
↓
Pipeline
↓
StandardScaler
↓
Logistic Regression
↓
Cross Validation
↓
Prediction
↓
Evaluation

---

## Pipeline Used

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

---

## Model Evaluation

Metrics Used:

* Cross Validation Accuracy
* Test Accuracy
* Classification Report

---

## Example Prediction

Input:

```python
{
    "hours_studied": 6,
    "attendance": 78,
    "assignments_completed": 7
}
```

Output:

```text
Student will PASS
```

---

## Learning Outcomes

Through this project I learned:

* Why Pipelines are important
* How StandardScaler works inside a Pipeline
* How to prevent Data Leakage
* How Cross Validation evaluates model performance
* How professional ML workflows are built
* How Logistic Regression can be integrated with Pipelines

---

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

## Project Status

✅ Completed

Part of my Machine Learning learning journey and ML Projects collection.
