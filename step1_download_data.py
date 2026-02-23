import yfinance as yf

print("Downloading stock data...")

data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

data.to_csv("stock.csv")

print("Download complete!")
print(data.head())