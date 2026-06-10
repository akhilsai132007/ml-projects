# Optimized Student Performance Predictor 🚀

A Machine Learning project that predicts whether a student will pass or fail based on study habits and academic activity.

## Dataset

The dataset contains:

- hours_studied
- attendance
- assignments_completed
- pass

Target column:

- pass
  - 0 = Fail
  - 1 = Pass

## Models Used

- Decision Tree Classifier
- Random Forest Classifier

## Techniques Used

- Train-Test Split
- Accuracy Score
- Hyperparameter Tuning
- GridSearchCV
- Cross Validation
- Model Comparison

## Project Workflow

Dataset
↓
Train-Test Split
↓
Train Decision Tree and Random Forest
↓
Check Accuracy Before Tuning
↓
Apply GridSearchCV
↓
Find Best Hyperparameters
↓
Evaluate Tuned Models
↓
Compare Before vs After Tuning

## Hyperparameters Tuned

### Decision Tree

- max_depth
- min_samples_split

### Random Forest

- n_estimators
- max_depth
- min_samples_split

## Goal

The goal of this project is to understand how hyperparameter tuning improves model performance and how GridSearchCV helps find the best model settings automatically.

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Output

The program displays:

- Decision Tree accuracy before tuning
- Best Decision Tree parameters
- Decision Tree accuracy after tuning
- Random Forest accuracy before tuning
- Best Random Forest parameters
- Random Forest accuracy after tuning
- Final comparison of all results

## Learning Outcome

Through this project, I learned how to:

- Train multiple ML models
- Tune hyperparameters using GridSearchCV
- Use Cross Validation inside Grid Search
- Compare model performance before and after tuning
- Select a better performing model