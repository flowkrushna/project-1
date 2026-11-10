import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # NAYA: AI ko save karne ke liye

print("Fetching data & applying Feature Engineering...\n")

ticker = "GLD"
gold = yf.Ticker(ticker)
data = gold.history(period="60d", interval="15m")

data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()

data['High-Low'] = data['High'] - data['Low']
data['High-PrevClose'] = np.abs(data['High'] - data['Close'].shift(1))
data['Low-PrevClose'] = np.abs(data['Low'] - data['Close'].shift(1))
data['TrueRange'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
data['ATR'] = data['TrueRange'].rolling(window=14).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
data['Hour'] = data.index.hour

data['Dist_to_EMA21'] = data['Close'] - data['EMA_21']
data['RSI_Slope'] = data['RSI'] - data['RSI'].shift(3)
data['Volume_Spike'] = data['Volume'] / data['Volume_MA_20']

data['Future_Close'] = data['Close'].shift(-4)
data['Target'] = np.where(data['Future_Close'] > (data['Close'] + (data['ATR'] * 0.5)), 1, 0)
data.dropna(inplace=True)

features = ['Dist_to_EMA21', 'RSI_Slope', 'Volume_Spike', 'ATR', 'Hour']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("Training the upgraded AI Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("-" * 40)
print(f"Upgraded AI Prediction Accuracy : {accuracy * 100:.2f}%")
print("-" * 40)

# NAYA: Trained model ko hamesha ke liye hard drive me save karna
model_filename = "ai_trading_brain.pkl"
joblib.dump(model, model_filename)
print(f"Success! AI Brain saved as '{model_filename}' 🚀")