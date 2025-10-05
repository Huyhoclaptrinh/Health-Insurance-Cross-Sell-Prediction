import pandas as pd, argparse, pathlib
from sklearn.model_selection import train_test_split

def load_clean(path):
    df = pd.read_csv(path)
    # Minimal canonical cleaning
    df['Vehicle_Age'] = df['Vehicle_Age'].replace(
        {'> 2 Years':'gt2','1-2 Year':'1to2','< 1 Year':'lt1'}
    )
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes':1,'No':0})
    df['Driving_License'] = df['Driving_License'].astype(int)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="health_insurance/train.csv")   # your dataset path
    p.add_argument("--outdir", default="data/processed")
    args = p.parse_args()

    df = load_clean(args.raw)
    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 70/15/15 with stratification on target
    tr, temp = train_test_split(df, test_size=0.30, stratify=df["Response"], random_state=42)
    va, te = train_test_split(temp, test_size=0.50, stratify=temp["Response"], random_state=42)

    tr.to_csv(out/"train.csv", index=False)
    va.to_csv(out/"valid.csv", index=False)
    te.to_csv(out/"test.csv", index=False)

    # tiny sanity print
    print("Splits saved to data/processed/")
    for name, part in [("train", tr), ("valid", va), ("test", te)]:
        print(name, "shape:", part.shape, "positive rate:", round(part['Response'].mean(), 4))
