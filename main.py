from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import math

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
    allow_headers=["*"],  # Allow all headers
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
    initial_json = test_df.to_dict(orient="records")

 
    model = load_model("LSTM_RUL.h5")
    model.summary()

    sequence_length = 50  
    sequence_cols = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = sequence_cols

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    test_df[sequence_cols[2:]] = scaler.fit_transform(test_df[sequence_cols[2:]])

  
    seq_array_test_last = [
        test_df[test_df["id"] == unit][sequence_cols[1:]].values[-sequence_length:]
        for unit in test_df["id"].unique()
        if len(test_df[test_df["id"] == unit]) >= sequence_length
    ]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    print(f"Shape of test sequences: {seq_array_test_last.shape}")


    predictions_scaled = model.predict(seq_array_test_last)
    print("Predictions generated successfully!")


    min_rul, max_rul = 0, 361
    predictions = predictions_scaled * (max_rul - min_rul) + min_rul

    unit_ids = test_df["id"].unique()[-len(predictions):]  # Match unit IDs with predictions
    prediction_df = pd.DataFrame({
        "unit_number": unit_ids,
        "Predicted_RUL": predictions.flatten()
    })
    

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

    health_df["Health Status"] = health_df["Predicted_RUL"].apply(classify_health)
    health_df["Next Maintenance Due"] = health_df["Predicted_RUL"].apply(schedule_maintenance)


    # Anomaly Detection
    iforest_model = joblib.load("multi_sensor_model.pkl")
    print("Model loaded successfully.")
    
    sequence_cols = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    df.columns = sequence_cols
    
    sensor_features = [f"s{i}" for i in range(1, 22) if f"s{i}" in df.columns]
    X = df[sensor_features]
    X.fillna(X.mean(), inplace=True) 
    
    df["anomaly"] = iforest_model.predict(X)
    
    df.to_csv("anomaly_results.csv", index=False)
    print("Anomaly results saved as anomaly_results.csv")
    
    """sensor_means = df[sensor_features].mean()
    sensor_stds = df[sensor_features].std()
    
    anomalies_detected = []
    
    for unit_id in df["id"].unique():
        unit_anomalies = df[df["id"] == unit_id]
        anomalous_sensors = []
    
        for sensor in sensor_features:
            if any(abs(unit_anomalies[sensor] - sensor_means[sensor]) > (3 * sensor_stds[sensor])):
                anomalous_sensors.append(sensor)
    
        anomalies_detected.append({"id": unit_id, "anomalous_sensors": anomalous_sensors})
    
    anomaly_report = pd.DataFrame(anomalies_detected)
    anomaly_report.to_csv("anomaly_report.csv", index=False)
    print("Anomaly report saved to anomaly_report.csv")
    
    #anomaly_df = pd.read_csv("anomaly_report.csv")

    """

    # Sanitize the final data
    json_data = health_df.to_dict(orient="records")
    json_anomaly = (anomaly_results.to_dict(orient="records"))

    return {"filename": latest_file, "original_data": initial_json, "final_data_RUL": json_data, "final_data_ANA": json_anomaly}
