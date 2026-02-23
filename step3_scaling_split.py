import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load CSV and skip first 2 bad rows
data = pd.read_csv("stock.csv", skiprows=2)

# Rename columns properly
data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Keep only Date and Close
data = data[["Date", "Close"]]

# Convert Date
data["Date"] = pd.to_datetime(data["Date"])

# Set Date as index
data.set_index("Date", inplace=True)

# Convert Close to numeric (VERY IMPORTANT)
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# Fill missing values
data.ffill(inplace=True)

# Scaling

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Save scaler for later use
import joblib
joblib.dump(scaler, "scaler.save")

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print("Scaling & split completed!")
print("Train size:", len(train_data))
print("Test size:", len(test_data))