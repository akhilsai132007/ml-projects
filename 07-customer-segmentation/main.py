import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv("data.csv")

X = df[
    [
        "age",
        "annual_income",
        "spending_score"
    ]
]

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_scaler)

    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaler)

df["cluster"] = clusters

print("\n===== CLUSTER METRICS =====")
print("Inertia:", kmeans.inertia_)

silhouette = silhouette_score(
    X_scaler,
    clusters
)

print("Silhouette Score:", round(silhouette, 4))

print("\n===== CLUSTER ANALYSIS =====")

print(
    df.groupby("cluster")[
        [
            "age",
            "annual_income",
            "spending_score"
        ]
    ].mean()
)

print("\n===== CUSTOMER CLUSTERS =====")
print(df)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["annual_income"],
    df["spending_score"],
    c=df["cluster"],
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()