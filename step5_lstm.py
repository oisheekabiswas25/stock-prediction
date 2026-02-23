import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Google stock data
data = pd.read_csv("goog.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data.ffill(inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences (lookback window = 5)
X, y = [], []
lookback = 5

for i in range(len(scaled_data) - lookback):
    X.append(scaled_data[i:i + lookback])
    y.append(scaled_data[i + lookback])

X = np.array(X)
y = np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save model and scaler for Flask app
model.save("lstm_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")

# Predict
pred = model.predict(X_test)

print("LSTM model trained and saved successfully!")