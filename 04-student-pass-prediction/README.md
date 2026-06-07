# Student Pass Prediction 🎯

A beginner Machine Learning Classification project that predicts whether a student will pass or fail based on study hours and attendance using Logistic Regression.

## Features

* Load student data from CSV
* Train/Test Split
* Train Logistic Regression Model
* Predict Pass/Fail
* Evaluate Model Accuracy
* Generate Confusion Matrix
* Generate Classification Report
* Accept User Input for Prediction
* Visualize Student Performance

---

## Dataset

The dataset contains:

| Column        | Description             |
| ------------- | ----------------------- |
| hours_studied | Number of hours studied |
| attendance    | Attendance percentage   |
| pass          | Yes = Pass, No = Fail   |

Example:

| hours_studied | attendance | pass |
| ------------- | ---------- | ---- |
| 5             | 90         | Yes  |
| 2             | 40         | No   |
| 8             | 95         | Yes  |

---

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-Learn

---

## Machine Learning Workflow

1. Load Dataset
2. Data Preparation
3. Train/Test Split
4. Train Logistic Regression Model
5. Predict Student Results
6. Evaluate Model Performance
7. Visualize Results

---

## Classification Concepts Used

### Binary Classification

The model predicts one of two classes:

* Yes (Pass)
* No (Fail)

### Logistic Regression

Used to classify students into Pass or Fail categories.

---

## Evaluation Metrics

### Accuracy Score

Measures how many predictions are correct.

### Confusion Matrix

Shows correct and incorrect predictions.

### Classification Report

Provides:

* Precision
* Recall
* F1 Score

---

## Visualization

### Scatter Plot

Shows the relationship between:

* Hours Studied
* Pass/Fail Status

---

## Project Structure

04-student-pass-prediction/

├── student_pass_data.csv

├── main.py

├── README.md

└── requirements.txt

---

## Example Output

Accuracy: 100%

Input:

Hours Studied: 6

Attendance: 85

Output:

Prediction: PASS ✅

---

## Learning Outcomes

Through this project, I learned:

* Classification
* Binary Classification
* Classes and Labels
* Logistic Regression
* Train/Test Split
* Accuracy Score
* Confusion Matrix
* Classification Report
* User Input Prediction
* Data Visualization

---

