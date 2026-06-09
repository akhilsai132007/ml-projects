# 🎯 Customer Segmentation using K-Means Clustering

A Machine Learning project that groups customers based on their age, annual income, and spending score using K-Means Clustering.

This project is built to understand Unsupervised Learning, customer grouping, cluster analysis, and clustering evaluation metrics.

---

## 📌 Project Overview

Customer Segmentation is used to divide customers into different groups based on similar behavior.

In this project, K-Means Clustering is used to create customer groups without using any target column.

---

## 📊 Dataset Columns

| Column | Description |
|--------|-------------|
| customer_id | Unique customer ID |
| age | Age of the customer |
| annual_income | Annual income of the customer |
| spending_score | Customer spending score |

---

## 🧠 Concepts Used

- Unsupervised Learning
- K-Means Clustering
- Feature Scaling
- StandardScaler
- Centroids
- Cluster Formation
- Elbow Method
- Inertia
- Silhouette Score
- Cluster Analysis
- Data Visualization

---

## 🛠 Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn

---

## 📈 Evaluation Metrics

### Inertia

Inertia measures how close data points are to their cluster centroid.

Lower inertia means the clusters are more compact.

### Silhouette Score

Silhouette Score measures how well-separated the clusters are.

Score range:

| Score | Meaning |
|-------|---------|
| Near 1 | Excellent clustering |
| Around 0.5 | Good clustering |
| Around 0 | Overlapping clusters |
| Negative | Poor clustering |

---

## 📊 Results

The model created 3 customer groups.

### Cluster 0

- Middle-aged customers
- Medium income
- Low spending score

### Cluster 1

- Higher income customers
- High spending score
- Premium customer segment

### Cluster 2

- Young customers
- Lower income
- Budget-conscious customers

---

## 📌 Model Performance

```text
Inertia: 21.7617
Silhouette Score: 0.5557