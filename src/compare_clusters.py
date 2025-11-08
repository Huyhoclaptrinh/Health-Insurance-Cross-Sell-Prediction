
import pandas as pd
from src.data_load import load_clean
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Load and clean data
df = load_clean("health_insurance/train.csv")

# Define feature lists
NUM = ["Age", "Annual_Premium", "Vintage"]
BIN = ["Driving_License", "Previously_Insured", "Vehicle_Damage"]
CAT = ["Vehicle_Age", "Region_Code", "Policy_Sales_Channel"]

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("bin", "passthrough", BIN)
    ],
    remainder="drop"
)

# --- Create and Save Final Model ---
print("--- Training and Saving Final 2-Cluster Model ---")

# Create the full pipeline with clustering
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=2, random_state=42, n_init=10))
])

# Fit the pipeline and get labels
labels = pipeline.fit_predict(df)
df['Cluster'] = labels

# Save results for visualization
df.to_csv("data/processed/clustered_data.csv", index=False)

print("Final clustered data saved to data/processed/clustered_data.csv")
