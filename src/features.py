import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from src.clustering import CustomerSegmentation

NUM = ["Age", "Annual_Premium", "Vintage"]
BIN = ["Driving_License", "Previously_Insured", "Vehicle_Damage"]
CAT_SMALL = ["Vehicle_Age"]                     # few levels: lt1, 1to2, gt2
CAT_BIG = ["Region_Code", "Policy_Sales_Channel"]  # many levels (anonymized)

def _winsorize(df: pd.DataFrame, q_low=0.01, q_high=0.99):
    df = df.copy()
    for c in NUM:
        lo, hi = df[c].quantile(q_low), df[c].quantile(q_high)
        df[c] = df[c].clip(lo, hi)
    return df

def _winsorize_transformer(X):
    return _winsorize(X)

def build_preprocessor():
    winsor = FunctionTransformer(_winsorize_transformer, validate=False)
    enc_small = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # Min-frequency one-hot keeps rare categories grouped to avoid huge sparse matrices
    enc_big = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    
    cluster_pipe = Pipeline([
        ('clustering', CustomerSegmentation(n_clusters=5, random_state=42)),
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM),
            ("bin", "passthrough", BIN),
            ("cat_small", enc_small, CAT_SMALL),
            ("cat_big", enc_big, CAT_BIG),
            ("cluster", cluster_pipe, NUM)
        ],
        remainder="drop",
        sparse_threshold=0.2  # return sparse if still efficient
    )
    return Pipeline([("winsor", winsor), ("ct", ct)])