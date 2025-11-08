
import pandas as pd
from src.data_load import load_clean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load and clean data
df = load_clean("health_insurance/train.csv")

# Select ONLY the original numerical features from the report
features = df[["Age", "Annual_Premium", "Vintage"]]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Take a 10% sample for scoring
sample_features = pd.DataFrame(scaled_features).sample(frac=0.1, random_state=42)

# --- K-Means Model ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(sample_features)
kmeans_score = silhouette_score(sample_features, kmeans_labels)

# --- Gaussian Mixture Model ---
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(sample_features)
gmm_score = silhouette_score(sample_features, gmm_labels)

# --- Print Scores ---
print("--- Baseline Model Scores ---")
print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
print(f"GMM Silhouette Score: {gmm_score:.4f}")
