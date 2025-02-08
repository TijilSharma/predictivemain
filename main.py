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

@app.api_route("/load-data", methods=["GET", "POST","HEAD"])
def load_data():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".txt")]
    if not files:
        return {"error": "No TXT files found"}

    latest_file = sorted(files)[-1]  # Get the latest uploaded file
    file_path = f"{UPLOAD_FOLDER}/{latest_file}"

    test_df = pd.read_csv(file_path, sep=" ", header=None)

    # Load the LSTM model
    model = load_model("LSTM_RUL.h5")

    # Summary of the model architecture
    model.summary()

    # Parameters
    sequence_length = 50  # Same as used during training
    sequence_cols = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

    
    # Drop extra columns and assign proper column names
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = sequence_cols

    # Normalize data if required (ensure test data matches preprocessing from training)
    # Assuming Min-Max Normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    test_df[sequence_cols[2:]] = scaler.fit_transform(test_df[sequence_cols[2:]])

    # Preprocess test data: Create sequences
    seq_array_test_last = [
        test_df[test_df["id"] == unit][sequence_cols[1:]].values[-sequence_length:]
        for unit in test_df["id"].unique()
        if len(test_df[test_df["id"] == unit]) >= sequence_length
    ]
    # Convert to NumPy array
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    print(f"Shape of test sequences: {seq_array_test_last.shape}")

    # Predict Remaining Useful Life (RUL)
    predictions_scaled = model.predict(seq_array_test_last)
    print("Predictions generated successfully!")

    # Scale predictions back to original range
    min_rul, max_rul = 0, 361
    predictions = predictions_scaled * (max_rul - min_rul) + min_rul

    # Prepare predictions for saving
    unit_ids = test_df["id"].unique()[-len(predictions):]  # Match unit IDs with predictions
    prediction_df = pd.DataFrame({
        "unit_number": unit_ids,
        "Predicted_RUL": predictions.flatten()
    })
    
    # Save predictions to CSV
    prediction_df.to_csv("predictions.csv", index=False)
    print("Predictions (scaled back) saved to predictions_scaled_back.csv!")

    health_df = pd.read_csv("predictions.csv")
    W1 = 122
    W0 = 47
    def classify_health(rul):
    if rul > W1:
        return "Low Risk"
    elif W0 <= rul <= W1:
        return "Medium Risk"
    else:
        return "High Risk"

    def schedule_maintenance(rul):
        if rul > W1:
            return "Maintenance in 30 days"
        elif W0 <= rul <= W1:
            return "Urgent: 5 days"
        else:
            return "Immediate Action Required!"

    health_df["Health Status"] = health_df["Predicted RUL"].apply(classify_health)
    health_df["Next Maintenance Due"] = health_df["Predicted RUL"].apply(schedule_maintenance)


   
    # Convert DataFrame to JSON
    json_data = health_df.to_dict(orient="records")

    return {"filename": latest_file, "data": json_data}
