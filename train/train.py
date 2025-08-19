
# train/train.py
# Usage: python train/train.py --csv data/scraply_price_training_v2.csv --out model/model.joblib
import argparse, json, os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

def build_pipeline(categorical, numeric):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="passthrough",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])
    return pipe

def main(args):
    df = pd.read_csv(args.csv)
    target_col = "FinalPrice" if "FinalPrice" in df.columns else "Final_Price" if "Final_Price" in df.columns else "FinalPrice"
    if target_col not in df.columns:
        # Try common names
        if "FinalPrice" in df.columns:
            target_col = "FinalPrice"
        elif "Final_Price" in df.columns:
            target_col = "Final_Price"
        else:
            raise ValueError("No target column named FinalPrice or Final_Price in CSV. Found: " + ", ".join(df.columns))
    cat_cols = ["Category","Brand","Condition","BodyType"]
    num_cols = ["ActualPrice","RecyclePossible","ReusePossible","YearsUsed","Running"]
    # Ensure columns exist
    for col in cat_cols + num_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col} in CSV")

    # Cast boolean-like columns to int
    for c in ["RecyclePossible","ReusePossible","Running"]:
        df[c] = df[c].astype(int)

    df["YearsUsed"] = df["YearsUsed"].astype(int)
    X = df[cat_cols + num_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(cat_cols, num_cols)
    print("Training model...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    try:
        mape = mean_absolute_percentage_error(y_test, preds)
    except Exception:
        mape = np.mean(np.abs((y_test - preds) / np.where(y_test==0, 1, y_test)))

    print(f"R2: {r2:.3f} | MAE: {mae:.2f} | MAPE: {mape*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print("Saved model to", args.out)

    # Save schema for client validation
    schema = {
        "categoricals": {
            "Category": sorted(df["Category"].dropna().unique().tolist()),
            "Brand": sorted(df["Brand"].dropna().unique().tolist()),
            "Condition": sorted(df["Condition"].dropna().unique().tolist()),
            "BodyType": sorted(df["BodyType"].dropna().unique().tolist()),
        },
        "numeric_ranges": {
            "ActualPrice": [float(df["ActualPrice"].min()), float(df["ActualPrice"].max())],
            "YearsUsed": [int(df["YearsUsed"].min()), int(df["YearsUsed"].max())]
        }
    }
    with open(os.path.join(os.path.dirname(args.out), "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    print("Saved schema.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/scraply_price_training_v2.csv")
    parser.add_argument("--out", default="model/model.joblib")
    args = parser.parse_args()
    main(args)
