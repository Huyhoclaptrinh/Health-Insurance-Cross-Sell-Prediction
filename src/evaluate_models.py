
import pandas as pd
import numpy as np
from src.data_load import load_clean
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif

# Load and clean data
df = load_clean("health_insurance/train.csv")

# --- Feature Selection ---
# Separate target variable
X = df.drop(["id", "Response"], axis=1)
y = df["Response"]

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create a preprocessor to handle categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)
processed_feature_names = numerical_features.tolist() + \
                          preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()

# Calculate mutual information scores
mi_scores = mutual_info_classif(X_processed, y, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=processed_feature_names)

# Select top 6 features
top_features = mi_scores_series.nlargest(6).index.tolist()
print("Selected features for clustering:", top_features)

# Get the data for the selected features
original_top_features = []
for feature in top_features:
    for original_cat in categorical_features:
        if feature.startswith(original_cat):
            if original_cat not in original_top_features:
                original_top_features.append(original_cat)
            break
    else:
        if feature not in original_top_features:
            original_top_features.append(feature)

features_df = df[original_top_features]

# Preprocess the selected features
cat_features_in_selection = [f for f in categorical_features if f in features_df.columns]
num_features_in_selection = [f for f in numerical_features if f in features_df.columns]

final_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features_in_selection),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_in_selection)
    ],
    remainder='passthrough'
)

scaled_features = final_preprocessor.fit_transform(features_df)

# Take a 10% sample for scoring
np.random.seed(42)
sample_indices = np.random.choice(scaled_features.shape[0], size=int(scaled_features.shape[0] * 0.1), replace=False)
sample_features = scaled_features[sample_indices]

# --- Hyperparameter Tuning for Number of Clusters ---
cluster_range = range(2, 11)
kmeans_scores = []
gmm_scores = []

print("\n--- Tuning Number of Clusters (2 to 10) ---")
for n_clusters in cluster_range:
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(sample_features)
    kmeans_score = silhouette_score(sample_features, kmeans_labels)
    kmeans_scores.append(kmeans_score)

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(sample_features)
    gmm_score = silhouette_score(sample_features, gmm_labels)
    gmm_scores.append(gmm_score)
    print(f"  - Tested {n_clusters} clusters: KMeans Score = {kmeans_score:.4f}, GMM Score = {gmm_score:.4f}")

# Find best scores
best_kmeans_score = max(kmeans_scores)
best_kmeans_clusters = cluster_range[np.argmax(kmeans_scores)]

best_gmm_score = max(gmm_scores)
best_gmm_clusters = cluster_range[np.argmax(gmm_scores)]

# --- Print Best Scores ---
print("\n--- Optimal Number of Clusters Found ---")
print(f"Best K-Means: {best_kmeans_clusters} clusters with Silhouette Score: {best_kmeans_score:.4f}")
print(f"Best GMM: {best_gmm_clusters} clusters with Silhouette Score: {best_gmm_score:.4f}")
