# task8_kmeans.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defensive settings to avoid loky/joblib subprocess probes on some Windows setups
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Try to use psutil if available (recommended)
try:
    import psutil
    print(f"psutil found — logical CPU count: {psutil.cpu_count(logical=True)}")
except Exception:
    print("psutil not installed. For fewer loky subprocess issues, run: pip install psutil")

# Clean output messages
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------- Configuration ----------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set the dataset path (put Mall_Customers.csv next to this script or give full path)
DATASET_PATH = "Mall_Customers.csv"

# Whether to run PCA for a 2D projection (useful if you cluster with >2 features)
USE_PCA_FOR_2D_VIS = False  # set True if you want PCA-based 2D plots
PCA_COMPONENTS = 2

# Range for Elbow method
MAX_K = 11
RANDOM_STATE = 42
# ------------------------------------------------

print("\n--- Task 8: Clustering with K-Means ---\n")

# 1. Load dataset
if not os.path.isfile(DATASET_PATH):
    print(f"Error: Dataset not found at '{DATASET_PATH}'.\nPlease put 'Mall_Customers.csv' in the script folder or update DATASET_PATH.")
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded. Shape: {df.shape}\nColumns: {list(df.columns)}\n")

# For Mall_Customers dataset we typically use: 'Annual Income (k$)', 'Spending Score (1-100)'
# If those columns exist, use them; else use numeric columns automatically.
expected_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
if all(col in df.columns for col in expected_cols):
    features = expected_cols.copy()
else:
    # fallback: select numeric columns (except ID if present)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # try to remove obvious ID headers
    for c in ['CustomerID', 'ID', 'Id', 'Index']:
        if c in numeric_cols:
            numeric_cols.remove(c)
    if len(numeric_cols) < 2:
        raise SystemExit("Not enough numeric columns to cluster. Please provide a suitable dataset.")
    features = numeric_cols[:2]  # pick first two numeric columns
print(f"Using features for clustering: {features}\n")

X = df[features].copy()

# 1b. Optional: show basic scatter (saved)
plt.figure(figsize=(7, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=50, alpha=0.7)
plt.title("Feature scatter (raw)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.grid(True, linestyle='--', alpha=0.5)
raw_scatter_path = os.path.join(OUTPUT_DIR, "raw_feature_scatter.png")
plt.tight_layout()
plt.savefig(raw_scatter_path)
plt.close()
print(f"Saved raw feature scatter to: {raw_scatter_path}")

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# If user wants PCA for 2D visualization when using >2 features:
if USE_PCA_FOR_2D_VIS and X_scaled.shape[1] > 2:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_vis = pca.fit_transform(X_scaled)
    print(f"PCA applied for 2D visualization (explained variance ratio: {pca.explained_variance_ratio_})")
else:
    X_vis = X_scaled[:, :2] if X_scaled.shape[1] >= 2 else X_scaled

# 3. Elbow Method to find optimal K
wcss = []
for k in range(1, MAX_K):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=RANDOM_STATE)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, MAX_K), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (WCSS vs. K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
elbow_path = os.path.join(OUTPUT_DIR, "elbow_method_plot.png")
plt.tight_layout()
plt.savefig(elbow_path)
plt.close()
print(f"Elbow plot saved to: {elbow_path}")

# Auto-pick: here we let the user choose; typical Mall dataset K=5 — we'll default to 5.
optimal_k = 5
print(f"Selected K = {optimal_k} (you can change 'optimal_k' variable if you want a different K)\n")

# 4. Fit KMeans and assign cluster labels
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=RANDOM_STATE)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# 5. Evaluate with Silhouette Score
if optimal_k > 1:
    sil_score = silhouette_score(X_scaled, labels)
else:
    sil_score = float('nan')
print(f"Silhouette Score for K={optimal_k}: {sil_score:.4f}\n")

# Save silhouette to a small txt file
with open(os.path.join(OUTPUT_DIR, "silhouette_score.txt"), "w") as f:
    f.write(f"Silhouette Score for K={optimal_k}: {sil_score:.6f}\n")

# 4b. Visualize clusters (using X_vis for plotting)
plt.figure(figsize=(10, 8))
# choose color map for up to 12 clusters
cmap = plt.get_cmap('tab10')
for i in range(optimal_k):
    pts = X_vis[labels == i]
    plt.scatter(pts[:, 0], pts[:, 1], s=90, alpha=0.7, label=f"Cluster {i}", edgecolors='w', linewidth=0.5, cmap=None)

# plot centroids projected to vis-space if PCA used, else use cluster centers' first two scaled dims
if USE_PCA_FOR_2D_VIS and X_scaled.shape[1] > 2:
    centroids_vis = pca.transform(kmeans.cluster_centers_)
else:
    centroids_vis = kmeans.cluster_centers_[:, :2]

plt.scatter(centroids_vis[:, 0], centroids_vis[:, 1], s=300, marker='*', label='Centroids', edgecolors='k')
plt.title(f'K-Means Clusters (K={optimal_k}) — {"PCA 2D" if USE_PCA_FOR_2D_VIS else "Scaled Features (first 2 dims)"}')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
cluster_plot_path = os.path.join(OUTPUT_DIR, f'kmeans_clusters_k{optimal_k}_plot.png')
plt.savefig(cluster_plot_path)
plt.close()
print(f"Cluster plot saved to: {cluster_plot_path}")

# Cluster summary (mean values)
cluster_summary = df.groupby('Cluster')[features + ['Age']] \
                   .mean().round(2) if 'Age' in df.columns else df.groupby('Cluster')[features].mean().round(2)
print("--- Cluster Interpretation (means) ---")
print(cluster_summary)
cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary_means.csv"))

# Save final DataFrame with cluster labels
final_csv_path = os.path.join(OUTPUT_DIR, "customer_segments.csv")
df.to_csv(final_csv_path, index=False)

print(f"\nAll outputs saved to folder: {OUTPUT_DIR}")
print("Files created:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)
print("\n--- Task complete. Run the script again to change K or visualize other features. ---\n")
