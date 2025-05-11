# Project Objective
 The goal of this project is to analyze customer preferences using the FoodMart dataset. Insights will improve recommendation systems and guide inventory management by identifying customer segments through clustering.
# 1. Data Import and Libraries
We imported essential libraries for data manipulation, visualization, preprocessing, clustering, and evaluation.
Libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, warnings
Dataset: StoresData.csv — Customer transaction and demographic data from FoodMart
# 2. Data Cleaning and Preprocessing
Actions Taken:
- Removed unnamed columns
- Dropped duplicate records
- Handled missing values:
    - Categorical columns: filled with mode
    - Numeric columns: filled with median
- Encoded categorical variables using Label Encoding
- Standardized all features using StandardScaler
Results:
- No missing values remained
- All columns transformed to numeric, suitable for clustering algorithms
# 3. Dimensionality Reduction (PCA)
Applied PCA to reduce features to 2 dimensions for visualization to simplify plotting and understanding of clusters.

# 4. Cluster Analysis
Optimal Cluster Determination
Used K-Medoids clustering with Silhouette Score to determine best k
k=4
Silhouette Score: 0.42
K-Medoids Clustering
- Applied K-Medoids algorithm with optimal k=4
- Silhouette Score: 0.4
Hierarchical Clustering
- Applied Agglomerative Clustering (Ward linkage)
- Silhouette Score: {hierarchical_score:.4f} (to be filled)
# 5. Exploratory Data Analysis (EDA)
Visualizations Created:
- Cluster Scatterplots (K-Medoids & Hierarchical)
- Silhouette Score Plot for k range 2–10
- Dendrogram to visualize hierarchical cluster structure
- Feature Histograms
- Cluster Distribution Pie Charts
- Boxplots to compare spending across clusters
# 6. Cluster Classification (Predictive Modeling)
Actions Taken:
- Split data into train/test sets (80/20)
- Trained Random Forest Classifier to predict clusters
Results:
- Accuracy: 84%
- Classification Report & Confusion Matrix: Available Below (Screenshots)
- Top Features: Available Below (Screenshots)


# 7. Conclusion & Business Implications
- Identified {4} customer segments with distinct characteristics
- Provided insights for targeted marketing and inventory management
- Built predictive model with accuracy {84%} to classify future customers into segments

Business Actions:
- Design personalized promotions based on cluster behavior
- Adjust inventory according to dominant segment preferences
# 8. Reproducibility
- Random seed fixed (random_state=42)
- All preprocessing steps documented and codified

# Screenshots:
![Screenshot01](https://github.com/hossam-ahmedd/Customer-Preferences-Analysis-/blob/main/Screenshot%202025-05-11%20033515.png?raw=true)
