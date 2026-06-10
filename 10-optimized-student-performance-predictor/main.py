import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df[
    [
        "hours_studied",
        "attendance",
        "assignments_completed"
    ]
]

y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print("Decision Tree Before Tuning:", dt_accuracy)

dt_params = {
    "max_depth": [2, 3, 4, 5, 6, 7],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_params,
    cv=5,
    scoring="accuracy"
)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

best_dt_pred = best_dt.predict(X_test)

best_dt_accuracy = accuracy_score(
    y_test,
    best_dt_pred
)

print("\nBest Decision Tree Parameters:")
print(dt_grid.best_params_)

print("Decision Tree After Tuning:", best_dt_accuracy)

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\nRandom Forest Before Tuning:", rf_accuracy)

rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring="accuracy"
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

best_rf_pred = best_rf.predict(X_test)

best_rf_accuracy = accuracy_score(
    y_test,
    best_rf_pred
)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

print("Random Forest After Tuning:", best_rf_accuracy)

print("\n===== FINAL COMPARISON =====")

print("Decision Tree Before Tuning:", dt_accuracy)
print("Decision Tree After Tuning:", best_dt_accuracy)

print("Random Forest Before Tuning:", rf_accuracy)
print("Random Forest After Tuning:", best_rf_accuracy)