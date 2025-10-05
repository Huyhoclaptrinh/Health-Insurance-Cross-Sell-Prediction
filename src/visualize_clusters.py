import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

INP = "data/processed/clustered_data.csv"
OUT = pathlib.Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)

# Load clustered data
df = pd.read_csv(INP)

# --- Visualize K-Means vs GMM --- 

# Scatter plot comparison
fig1, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
sns.scatterplot(data=df, x="Age", y="Annual_Premium", hue="KMeans_Cluster", palette="viridis", alpha=0.6, ax=axes[0])
axes[0].set_title("Baseline: K-Means Segments")
sns.scatterplot(data=df, x="Age", y="Annual_Premium", hue="GMM_Cluster", palette="viridis", alpha=0.6, ax=axes[1])
axes[1].set_title("Improved: GMM Segments")
plt.suptitle("Comparison of Customer Segments (Age vs. Annual Premium)")
fig1.savefig(OUT/"cluster_comparison_scatterplot.png", bbox_inches="tight"); plt.close(fig1)

# Boxplot comparison for numerical features
for col in ["Age", "Annual_Premium", "Vintage"]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.boxplot(data=df, x="KMeans_Cluster", y=col, ax=axes[0])
    axes[0].set_title(f'{col} by K-Means Cluster')
    sns.boxplot(data=df, x="GMM_Cluster", y=col, ax=axes[1])
    axes[1].set_title(f'{col} by GMM Cluster')
    plt.suptitle(f"Comparison of {col} by Cluster")
    fig.savefig(OUT/f"cluster_comparison_boxplot_{col.lower()}.png", bbox_inches="tight"); plt.close(fig)

print("Cluster comparison visualization figures written to reports/figures/")