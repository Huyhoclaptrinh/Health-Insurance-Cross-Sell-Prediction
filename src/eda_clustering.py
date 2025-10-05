
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
