import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
from src.data_load import load_clean

INP = "health_insurance/train.csv"
OUT = pathlib.Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)

# Load and clean data
df = load_clean(INP)

# Select numerical features for clustering
num_features = df[["Age", "Annual_Premium", "Vintage"]]

# Create a pair plot
fig = sns.pairplot(num_features)
plt.suptitle("Pair Plot of Clustering Features", y=1.02)
fig.savefig(OUT/"clustering_features_pairplot.png", bbox_inches="tight")
plt.close()

print("Clustering EDA figure written to reports/figures/")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure df and load_clean are available
# The df variable should be available from previous cells, but we'll reload and clean for robustness
if 'df' not in locals():
    df = load_clean("health_insurance/train.csv")

# Define the key predictors
key_predictors = ['Vehicle_Damage', 'Previously_Insured', 'Driving_License']

print("--- EDA Figures for Key Behavioral Predictors ---")

# Figure 1: Individual Distributions of Key Predictors
plt.figure(figsize=(15, 5))
for i, feature in enumerate(key_predictors):
    plt.subplot(1, 3, i + 1)
    sns.countplot(data=df, x=feature, hue=feature, palette='viridis', legend=False)
    plt.title(f'Distribution of {feature.replace("_", " ")}')
    plt.xlabel(feature.replace("_", " "))
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUT/"key_predictors_distribution.png", bbox_inches="tight")
plt.close()

# Figure 2: Key Predictors in Relation to Response Variable
plt.figure(figsize=(15, 5))
for i, feature in enumerate(key_predictors):
    plt.subplot(1, 3, i + 1)
    sns.countplot(data=df, x=feature, hue='Response', palette='viridis')
    plt.title(f'{feature.replace("_", " ")} by Response')
    plt.xlabel(feature.replace("_", " "))
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No Insurance', 'Wants Insurance'])
plt.tight_layout()
plt.savefig(OUT/"key_predictors_by_response.png", bbox_inches="tight")
plt.close()

# Optional: Further insights with crosstabulations
print("\n--- Crosstabulations of Key Predictors with Response ---")
for feature in key_predictors:
    print(f"\nCrosstab for {feature.replace('_', ' ')} and Response:")
    print(pd.crosstab(df[feature], df['Response'], normalize='index').round(4))

print("EDA figures generated.")