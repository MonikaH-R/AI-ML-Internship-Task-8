# AI-ML-Internship-Task-8

 **Objective**

The main goal of this task is to perform **Unsupervised Learning** using the **K-Means Clustering Algorithm**.  
We apply K-Means to the *Mall Customer Segmentation Dataset* to identify distinct customer groups based on their spending patterns.


 **Tools & Libraries Used**
 
- **Python 3.x**
- **Scikit-learn** – for K-Means, scaling, and metrics  
- **Pandas** – for data loading and processing  
- **Matplotlib / Seaborn** – for visualization  
- **NumPy** – for numerical operations


##  Steps Followed

### **Step 1: Load and Visualize Dataset**
- Imported the **Mall Customer Segmentation Dataset** (`Mall_Customers.csv`).
- Displayed basic statistics and sample data.
- Created a scatter plot of selected features (e.g., Annual Income vs Spending Score).
- Optionally used **PCA** for 2D visualization.

### **Step 2: Fit K-Means and Assign Cluster Labels**
- Applied the **K-Means algorithm** from Scikit-learn.
- Standardized features using `StandardScaler` for accurate distance measurement.
- Assigned each data point to a cluster and stored labels in the dataset.

### **Step 3: Elbow Method to Find Optimal K**
- Computed the **Within-Cluster Sum of Squares (WCSS)** for K = 1 to 10.
- Plotted the **Elbow Curve** to visually identify the optimal number of clusters.

### **Step 4: Visualize Clusters**
- Created colorful 2D scatter plots with centroids highlighted.
- Visualized customer groups clearly using distinct color coding.

### **Step 5: Evaluate Clustering using Silhouette Score**
- Calculated the **Silhouette Score** to measure cluster quality.
- Higher Silhouette Score indicates better-defined clusters.



##  Outputs Generated
All output files are automatically saved inside the **`output/`** folder:
| File Name | Description |
|------------|-------------|
| `raw_feature_scatter.png` | Original dataset scatter plot |
| `elbow_method_plot.png` | Elbow Method curve |
| `kmeans_clusters_k5_plot.png` | Cluster visualization for K=5 |
| `silhouette_score.txt` | Silhouette score value |
| `customer_segments.csv` | Original dataset with cluster labels |
| `cluster_summary_means.csv` | Average feature values per cluster |



##  How to Run the Project
1. Open **PyCharm** or any Python IDE.  
2. Place `Mall_Customers.csv` in your project folder.  
3. Run the Python script:  
   ```bash
   python kmeans_clustering.py
