import pandas as pd

# Load Google stock CSV
data = pd.read_csv("goog.csv", index_col=0)

# Rename index to Date
data.index.name = "Date"

# Keep only Close column
data = data[['Close']]

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Save cleaned data
data.to_csv("stock_clean.csv")

print("Preprocessing done successfully!")
print(data.head())