# 1. Objective
# Analyze customer preferences using the FoodMart dataset for better recommendations and inventory management.

# 2. Import Libraries
import matplotlib
matplotlib.use('TkAgg')  # Fix for PyCharm plotting issue

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# 3. Load Data
df = pd.read_csv("D:\\Downloads\\Chrome Downloads\\StoresData.csv")

# 4. Data Cleaning and Preprocessing
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Initial Data Info:\n", df.info())
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
missing_cols = df.columns[df.isnull().any()]
for col in missing_cols:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Check all numeric before scaling
assert all(np.issubdtype(dtype, np.number) for dtype in df.dtypes), "Non-numeric column exists after encoding."

# Standardize the data
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# 5. PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

# 6. Find Optimal Clusters
def find_optimal_clusters(data, max_clusters=10):
    scores = []
    for k in range(2, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=k, random_state=42)
        labels = kmedoids.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    optimal_k = np.argmax(scores) + 2
    return optimal_k, scores

optimal_k, silhouette_scores = find_optimal_clusters(X_pca)
print(f"Optimal number of clusters: {optimal_k}")

# 7. Apply K-Medoids
kmedoids = KMedoids(n_clusters=optimal_k, random_state=42)
kmedoids_clusters = kmedoids.fit_predict(X_pca)
df['KMedoids_Cluster'] = kmedoids_clusters
kmedoids_score = silhouette_score(X_pca, kmedoids_clusters)
print(f"K-Medoids Silhouette Score: {kmedoids_score:.4f}")

# 8. Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
hierarchical_clusters = hierarchical.fit_predict(X_pca)
df['Hierarchical_Cluster'] = hierarchical_clusters
hierarchical_score = silhouette_score(X_pca, hierarchical_clusters)
print(f"Hierarchical Clustering Silhouette Score: {hierarchical_score:.4f}")

# 9. Visualizations
plt.figure(figsize=(18, 6))

# K-Medoids Plot
plt.subplot(1, 3, 1)
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
for i in range(optimal_k):
    plt.scatter(X_pca[kmedoids_clusters == i, 0], X_pca[kmedoids_clusters == i, 1],
                s=100, c=colors[i % len(colors)], label=f'Cluster {i + 1}')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1],
            s=300, c='yellow', marker='*', label='Medoids')
plt.title(f'K-Medoids Clustering (k={optimal_k})\nSilhouette: {kmedoids_score:.2f}')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

# Hierarchical Plot
plt.subplot(1, 3, 2)
for i in range(optimal_k):
    plt.scatter(X_pca[hierarchical_clusters == i, 0], X_pca[hierarchical_clusters == i, 1],
                s=100, c=colors[i % len(colors)], label=f'Cluster {i + 1}')
plt.title(f'Hierarchical Clustering (k={optimal_k})\nSilhouette: {hierarchical_score:.2f}')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

# Silhouette Score Plot
plt.subplot(1, 3, 3)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-Medoids')
plt.axvline(x=optimal_k, color='red', linestyle='--')
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Dendrogram
plt.figure(figsize=(12, 6))
plt.title("Dendrogram for Hierarchical Clustering")
Z = linkage(X_pca, method='ward')
dendrogram(Z)
plt.axhline(y=20, color='r', linestyle='--')  # Adjust as needed
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# 11. Histogram
df.hist(figsize=(12, 8))
plt.suptitle("Histogram of Dataset Features")
plt.tight_layout()
plt.show()

# 12. Pie Charts for Cluster Distribution
plt.figure(figsize=(12, 5))

# K-Medoids Cluster Distribution
plt.subplot(1, 2, 1)
kmedoids_counts = df['KMedoids_Cluster'].value_counts().sort_index()
plt.pie(kmedoids_counts, labels=[f'Cluster {i}' for i in kmedoids_counts.index],
        autopct='%1.1f%%', startangle=90, colors=colors[:len(kmedoids_counts)])
plt.title('K-Medoids Cluster Distribution')

# Hierarchical Cluster Distribution
plt.subplot(1, 2, 2)
hierarchical_counts = df['Hierarchical_Cluster'].value_counts().sort_index()
plt.pie(hierarchical_counts, labels=[f'Cluster {i}' for i in hierarchical_counts.index],
        autopct='%1.1f%%', startangle=90, colors=colors[:len(hierarchical_counts)])
plt.title('Hierarchical Cluster Distribution')

plt.tight_layout()
plt.show()

# 13. Boxplots (if total_spent exists)
if 'total_spent' in df.columns:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x='KMedoids_Cluster', y='total_spent', data=df)
    plt.title("Total Spending per K-Medoids Cluster")

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Hierarchical_Cluster', y='total_spent', data=df)
    plt.title("Total Spending per Hierarchical Cluster")

    plt.tight_layout()
    plt.show()

# 14. Train/Test Split
features = df.drop(columns=['KMedoids_Cluster', 'Hierarchical_Cluster'])
target = df['KMedoids_Cluster']  # or use Hierarchical_Cluster

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

print("Train and test sets created:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 15. Evaluation Metric using Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Evaluation Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))