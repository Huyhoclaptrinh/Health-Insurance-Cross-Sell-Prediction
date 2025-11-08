
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from src.data_load import load_clean
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# --- 1. Re-create the Final Model ---

# Load and clean data
df = load_clean("health_insurance/train.csv")

# Feature Selection
X = df.drop(["id", "Response"], axis=1)
y = df["Response"]
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)
processed_feature_names = numerical_features.tolist() + \
                          preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()

mi_scores = mutual_info_classif(X_processed, y, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=processed_feature_names)
top_features = mi_scores_series.nlargest(6).index.tolist()

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

# Train the 4-cluster KMeans model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("Successfully trained a 4-cluster model based on visual analysis.")

# --- 2. Create Visualization ---

# Use PCA to reduce dimensions for plotting
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

df['pca1'] = reduced_features[:, 0]
df['pca2'] = reduced_features[:, 1]

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='Cluster', palette='viridis', s=50, alpha=0.7)
plt.title('Customer Segments Visualized with PCA (4 Clusters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)

# Save the figure
OUT = pathlib.Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
fig_path = OUT/"final_4_cluster_scatterplot.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()

print(f"\nCluster visualization graph saved to: {fig_path}")

# --- 3. Interpret Principal Components ---

print("\n--- Interpreting the Principal Components ---")
# Get feature names after one-hot encoding
ohe_feature_names = final_preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features_in_selection)
all_feature_names = num_features_in_selection + ohe_feature_names.tolist()

# Create a DataFrame of PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=all_feature_names)

print("\n**Principal Component 1 (PC1):**")
print("Top 3 Positive Drivers:")
print(loadings['PC1'].sort_values(ascending=False).head(3))
print("\nTop 3 Negative Drivers:")
print(loadings['PC1'].sort_values(ascending=True).head(3))

print("\n**Principal Component 2 (PC2):**")
print("Top 3 Positive Drivers:")
print(loadings['PC2'].sort_values(ascending=False).head(3))
print("\nTop 3 Negative Drivers:")
print(loadings['PC2'].sort_values(ascending=True).head(3))


# --- 4. Define and Profile Each Cluster ---

print("\n--- Cluster Profiles ---")
# Use the original features for profiling, as they are more interpretable
profile_features = original_top_features
# Calculate the mode for categorical features and mean for numerical/binary
agg_funcs = {col: (lambda x: x.mode()[0]) for col in cat_features_in_selection}
agg_funcs.update({col: 'mean' for col in num_features_in_selection})
# Add Driving_License to the aggregation if it was selected
if 'Driving_License' in profile_features and 'Driving_License' not in num_features_in_selection:
    agg_funcs['Driving_License'] = 'mean'


cluster_profiles = df.groupby('Cluster').agg(agg_funcs)


# Rename columns for clarity
cluster_profiles.rename(columns={
    'Previously_Insured': 'Previously_Insured_Rate',
    'Vehicle_Damage': 'Vehicle_Damage_Rate',
    'Driving_License': 'Driving_License_Rate'
}, inplace=True)

# Print the profiles
for cluster_id, profile in cluster_profiles.iterrows():
    print(f"\n----- Cluster {cluster_id} -----")
    for feature, value in profile.items():
        if isinstance(value, float):
            print(f"  - {feature}: {value:.2f}")
        else:
            print(f"  - {feature}: {value}")
