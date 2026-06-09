# 🚀 Student Performance Model Comparison

A Machine Learning project that compares multiple classification models to determine the most reliable model for predicting student success.

This project evaluates Logistic Regression, Decision Tree, and Random Forest using Accuracy Score and Cross Validation Score.

---

# 📌 Project Overview

The goal of this project is to identify which Machine Learning model performs best for predicting whether a student will pass or fail based on study habits and academic performance.

The comparison is based on:

- Accuracy Score
- Cross Validation Score
- Model Reliability

---

# 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| hours_studied | Number of study hours |
| attendance | Attendance percentage |
| assignments_completed | Number of assignments completed |
| pass | Target variable (0 = Fail, 1 = Pass) |

---

# 🎯 Project Workflow

```text
Student Data
      ↓
Train/Test Split
      ↓
Train Models
      ↓
Predictions
      ↓
Accuracy Score
      ↓
Cross Validation
      ↓
Model Comparison
      ↓
Best Model Selection
```

---

# 🤖 Models Compared

## 1️⃣ Logistic Regression

Used for binary classification problems.

### Concepts Used

- Logistic Regression
- Classification
- Probability Prediction

---

## 2️⃣ Decision Tree

Creates a tree of decisions to classify data.

### Concepts Used

- Decision Tree Classifier
- Tree-Based Learning
- Classification

---

## 3️⃣ Random Forest

Uses multiple decision trees and combines their predictions.

### Concepts Used

- Random Forest Classifier
- Ensemble Learning
- Classification

---

# 📈 Evaluation Metrics

## Accuracy Score

Measures how many predictions are correct.

```text
Correct Predictions
-------------------
Total Predictions
```

---

## Cross Validation Score

Measures average model performance across multiple train-test splits.

Used to determine model reliability.

---

# 📊 Project Results

| Model | Accuracy | CV Score |
|---------|---------|---------|
| Logistic Regression | 100.00% | 91.79% |
| Decision Tree | 92.31% | 93.46% |
| Random Forest | 84.62% | 90.13% |

---

# 🏆 Best Model

## Decision Tree

### Why?

```text
Highest Cross Validation Score

93.46%
```

This indicates that the Decision Tree was the most reliable model on this dataset.

---

# 📉 Visualization

The project includes a bar chart comparing:

- Accuracy Score
- Cross Validation Score

for all models.

---

# 🛠 Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn

---

# 📚 Skills Practiced

- Classification
- Model Comparison
- Cross Validation
- Accuracy Evaluation
- Data Visualization
- Decision Making Using Metrics
- Reliability Analysis

---

# 📁 Project Structure

```text
09-student-performance-model-comparison/

├── data.csv
├── main.py
├── README.md
└── requirements.txt
```

---

# 🚀 How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

# 💡 Key Takeaway

A model with the highest Accuracy is not always the most reliable model.

Cross Validation provides a more trustworthy evaluation by testing the model on multiple data splits.

In this project, the Decision Tree achieved the highest Cross Validation Score and was selected as the most reliable model.

---

# 👨‍💻 Author

**Akhil Sai (Captain)**

CSE'28 Student

Focused on:

- Backend Development
- Machine Learning
- Data Analytics
- Open Source
- Real-World Projects
- GitHub Portfolio Building

---

# ⭐ Project Status

✅ Completed

✅ Model Comparison Implemented

✅ Cross Validation Applied

✅ Best Model Identified

✅ Visualization Added