import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load and clean Google stock data
data = pd.read_csv("goog.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data.ffill(inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create dataset (lag = 1 day)
X, y = [], []
for i in range(len(scaled) - 1):
    X.append(scaled[i])
    y.append(scaled[i + 1])

X = np.array(X)
y = np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)

# Save model for later use
import joblib
joblib.dump(model, "linear_model.save")

# Save test data for backend prediction/accuracy
joblib.dump((X_test, y_test), "linear_test.save")

# Predict
pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Linear Regression Results")
print("MSE:", mse)
print("R2 Score:", r2)