import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    results.append(
        {
            "Model": name,
            "Accuracy": round(accuracy * 100, 2),
            "CV Score": round(cv_scores.mean() * 100, 2)
        }
    )

comparison_df = pd.DataFrame(results)

print("\n===== STUDENT PERFORMANCE MODEL COMPARISON =====")
print(comparison_df)

best_model = comparison_df.sort_values(
    by="CV Score",
    ascending=False
).iloc[0]

print("\n===== BEST MODEL =====")
print("Best Model:", best_model["Model"])
print("Best CV Score:", best_model["CV Score"], "%")

plt.figure(figsize=(9, 5))

x = range(len(comparison_df))

plt.bar(
    [i - 0.2 for i in x],
    comparison_df["Accuracy"],
    width=0.4,
    label="Accuracy"
)

plt.bar(
    [i + 0.2 for i in x],
    comparison_df["CV Score"],
    width=0.4,
    label="CV Score"
)

plt.xticks(
    x,
    comparison_df["Model"]
)

plt.title("Student Performance Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score (%)")

plt.ylim(0, 110)

plt.legend()
plt.show()