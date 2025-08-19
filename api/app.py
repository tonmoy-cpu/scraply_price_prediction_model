
# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib, os, numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "model.joblib"))
pipe = None
if os.path.exists(MODEL_PATH):
    pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Scraply Price Prediction API", version="1.0.0")

class PredictIn(BaseModel):
    Category: str
    Brand: str
    Condition: str
    BodyType: str
    ActualPrice: float = Field(..., ge=0)
    RecyclePossible: bool
    ReusePossible: bool
    YearsUsed: int = Field(..., ge=0, le=50)
    Running: bool

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": pipe is not None}

@app.post("/predict")
def predict(item: PredictIn):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not available. Train model first and place model.joblib in the model/ directory.")
    df = pd.DataFrame([item.dict()])
    # convert bools to ints for model
    for col in ["RecyclePossible","ReusePossible","Running"]:
        df[col] = df[col].astype(int)
    # Predict
    try:
        y_hat = float(pipe.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Interval using trees if available
    lo, hi = None, None
    try:
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        X = pre.transform(df)
        tree_preds = np.vstack([est.predict(X) for est in model.estimators_])
        lo = float(np.percentile(tree_preds, 10))
        hi = float(np.percentile(tree_preds, 90))
    except Exception:
        lo, hi = max(0.0, y_hat*0.85), y_hat*1.15
    return {"predicted_price": round(y_hat,2), "range_10_90": [round(lo,2), round(hi,2)]}
