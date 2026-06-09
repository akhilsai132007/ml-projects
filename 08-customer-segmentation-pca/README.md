# 🚀 Customer Segmentation with PCA

A Machine Learning project that combines Principal Component Analysis (PCA) and K-Means Clustering to group customers based on their behavior.

This project demonstrates how dimensionality reduction can simplify data while preserving important information for clustering.

---

# 📌 Project Overview

Customer Segmentation is a popular Unsupervised Learning technique used to identify groups of similar customers.

In this project:

* Customer data is scaled using StandardScaler
* PCA is applied to reduce dimensions
* K-Means Clustering is performed on the reduced dataset
* Clustering performance is evaluated and compared

---

# 📊 Dataset Features

| Feature        | Description                |
| -------------- | -------------------------- |
| customer_id    | Unique Customer Identifier |
| age            | Customer Age               |
| annual_income  | Annual Income              |
| spending_score | Spending Behavior Score    |

---

# 🎯 Project Workflow

```text
Customer Data
      ↓
Feature Selection
      ↓
StandardScaler
      ↓
PCA
      ↓
Principal Components
      ↓
K-Means Clustering
      ↓
Customer Groups
      ↓
Cluster Analysis
```

---

# 🧠 Concepts Used

## Unsupervised Learning

Learning patterns from data without a target column.

## PCA (Principal Component Analysis)

Reduces the number of features while preserving most of the important information.

## Feature Reduction

Converts multiple features into fewer principal components.

## Principal Components

New features created by PCA that capture the most information in the dataset.

## K-Means Clustering

Groups similar customers into clusters.

## StandardScaler

Scales all features before PCA and K-Means.

## Cluster Analysis

Analyzes customer behavior within each cluster.

---

# 📈 PCA Results

### Explained Variance Ratio

```text
PC1 = 84.16%
PC2 = 13.78%
```

### Information Preserved

```text
97.94%
```

This means PCA reduced the dataset to 2 dimensions while preserving almost all important information.

---

# 📊 Clustering Results

### Inertia

```text
31.59
```

### Silhouette Score

```text
0.5828
```

The Silhouette Score indicates good cluster separation and clustering quality.

---

# 👥 Customer Segments Identified

## Cluster 0

* Young Customers
* Very High Income
* Very High Spending

### Segment Type

```text
Premium Customers
```

---

## Cluster 1

* Young to Middle-Aged Customers
* Low to Medium Income
* Low to Medium Spending

### Segment Type

```text
Budget Customers
```

---

## Cluster 2

* Middle-Aged Customers
* High Income
* High Spending

### Segment Type

```text
Loyal High-Value Customers
```

---

# 🛠 Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-Learn

---

# 📚 Important Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

---

# 🎯 Learning Outcomes

After completing this project, I learned:

* Principal Component Analysis (PCA)
* Feature Reduction
* Variance Concept
* Principal Components
* Explained Variance Ratio
* StandardScaler
* K-Means Clustering
* Cluster Analysis
* Dimensionality Reduction
* PCA + K-Means Integration

---

# 📁 Project Structure

```text
08-customer-segmentation-pca/

├── customer_data.csv
├── main.py
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

# 📊 Key Takeaway

PCA successfully reduced the dataset from three features to two principal components while preserving 97.94% of the original information. K-Means Clustering on the PCA-transformed data produced meaningful customer segments with a Silhouette Score of 0.5828.

---

# 👨‍💻 Author

**Akhil Sai (Captain)**

CSE'28 Student

Backend Development • Machine Learning • Data Analytics • Open Source

---

# ⭐ Project Status

Completed as part of my Machine Learning learning journey and GitHub portfolio.
