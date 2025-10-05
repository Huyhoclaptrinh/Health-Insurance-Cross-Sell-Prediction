
import pandas as pd
import numpy as np
from src.data_load import load_clean
from src.clustering import CustomerSegmentation
from src.gmm_clustering import CustomerSegmentationGMM
from sklearn.metrics import silhouette_score

# Load and clean data
df = load_clean("health_insurance/train.csv")
num_features = df[["Age", "Annual_Premium", "Vintage"]]

# --- Baseline Model: K-Means ---
print("Running Baseline Model: K-Means")
kmeans_model = CustomerSegmentation(n_clusters=5, random_state=42)
kmeans_labels = kmeans_model.fit_transform(num_features).flatten()
df["KMeans_Cluster"] = kmeans_labels

# --- Improved Model: GMM ---
print("Running Improved Model: GMM")
gmm_model = CustomerSegmentationGMM(n_clusters=5, random_state=42)
gmm_labels = gmm_model.fit_transform(num_features).flatten()
df["GMM_Cluster"] = gmm_labels

# --- Evaluate and Compare ---
print("\n--- Evaluation Metrics ---")
# Note: Silhouette score can be slow on large datasets. We'll use a sample.
sample_frac = 0.1
sample_df = num_features.sample(frac=sample_frac, random_state=42)
sample_kmeans_labels = kmeans_model.transform(sample_df).flatten()
sample_gmm_labels = gmm_model.transform(sample_df).flatten()

kmeans_silhouette = silhouette_score(sample_df, sample_kmeans_labels)
gmm_silhouette = silhouette_score(sample_df, sample_gmm_labels)

print(f"Baseline (K-Means) Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Improved (GMM) Silhouette Score: {gmm_silhouette:.4f}")

# Save results for visualization
df.to_csv("data/processed/clustered_data.csv", index=False)
print("\nClustered data saved to data/processed/clustered_data.csv")
