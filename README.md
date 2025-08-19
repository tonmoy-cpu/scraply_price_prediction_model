
# Scraply Price Service - Random Forest

This bundle contains a ready-to-run RandomForest training script and a FastAPI prediction service.
You uploaded a dataset and it has been copied into `data/scraply_price_training_v2.csv`.

## Files in this bundle
- `data/scraply_price_training_v2.csv` -- your uploaded dataset (used for training)
- `train/train.py` -- training script that creates a sklearn Pipeline and saves `model.joblib`
- `api/app.py` -- FastAPI app serving `/predict` endpoint (loads model from `model/model.joblib`)
- `model/` -- target folder to hold the trained `model.joblib` and `schema.json`
- `requirements.txt` -- Python dependencies

## Quickstart (Google Colab recommended)

### Option A - Google Colab (no local setup)
1. Open a new Colab notebook.
2. Upload this entire bundle zip or mount Google Drive and copy the files.
3. Run:
   ```bash
   !pip install -r requirements.txt
   !python train/train.py --csv data/scraply_price_training_v2.csv --out model/model.joblib
   ```
4. Start the API:
   ```bash
   !pip install "uvicorn[standard]"
   !uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

### Option B - Local
1. Create & activate a virtualenv with Python 3.10+
2. `pip install -r requirements.txt`
3. Train: `python train/train.py --csv data/scraply_price_training_v2.csv --out model/model.joblib`
4. Serve: `uvicorn api.app:app --reload`

## Example request
POST /predict with JSON body:
{
  "Category":"Laptop",
  "Brand":"Dell",
  "Condition":"Good",
  "BodyType":"Metal",
  "ActualPrice":40000,
  "RecyclePossible":true,
  "ReusePossible":true,
  "YearsUsed":2,
  "Running":true
}

Response:
{
  "predicted_price": 11500.23,
  "range_10_90": [10234.12, 12890.77]
}

## Next steps for MERN integration
- Deploy this service to Render/Railway/Heroku (follow README in repo)
- Create an Express route to proxy requests from your frontend to this API (example provided in previous instructions)

