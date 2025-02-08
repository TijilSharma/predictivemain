from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

app = FastAPI()
UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


from fastapi.middleware.cors import CORSMiddleware


# Allow requests from localhost and your frontend
origins = [
    "http://localhost:5173",
    "https://your-deployed-frontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "API is working!"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    
    # Save file temporarily
    with open(file_location, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename, "location": file_location}

@app.api_route("/load-data", methods=["GET", "POST", "HEAD"])
def load_data():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".txt")]
    if not files:
        return {"error": "No TXT files found"}

    latest_file = sorted(files)[-1]  # Get the latest uploaded file
    file_path = f"{UPLOAD_FOLDER}/{latest_file}"

    # ✅ Send original data first
    test_df = pd.read_csv(file_path, sep=" ", header=None)
    original_data_json = test_df.to_dict(orient="records")

    # Load the LSTM model
    model = load_model("LSTM_RUL.h5")
    model.summary()

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    test_df[sequence_cols[2:]] = scaler.fit_transform(test_df[sequence_cols[2:]])

    sequence_length = 50
    seq_array_test_last = [
        test_df[test_df["id"] == unit][sequence_cols[1:]].values[-sequence_length:]
        for unit in test_df["id"].unique()
        if len(test_df[test_df["id"] == unit]) >= sequence_length
    ]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    predictions = model.predict(seq_array_test_last)
    
    min_rul, max_rul = 0, 361
    predictions = predictions * (max_rul - min_rul) + min_rul

    unit_ids = test_df["id"].unique()[-len(predictions):]
    prediction_df = pd.DataFrame({
        "unit_number": unit_ids,
        "Predicted_RUL": predictions.flatten()
    })

    prediction_df.to_csv("predictions.csv", index=False)

    df_final = pd.read_csv("predictions.csv")
    predictions_json = df_final.to_dict(orient="records")

    return {
        "filename": latest_file,
        "original_data": original_data_json,  # ✅ First, send original data
        "predictions": predictions_json      # ✅ Then, send predictions
    }
