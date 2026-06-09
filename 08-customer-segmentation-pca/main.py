import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_pca)

df["cluster"] = clusters

print("\n===== PCA INFORMATION =====")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Information Preserved:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

print("\n===== PCA CLUSTER METRICS =====")
print("Inertia:", kmeans.inertia_)

silhouette = silhouette_score(X_pca, clusters)
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
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=100
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation with PCA")

plt.show()