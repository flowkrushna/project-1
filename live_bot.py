import yfinance as yf
import pandas as pd
import numpy as np
import joblib

print("🧠 Waking up the Gold AI Brain...")
model = joblib.load("ai_trading_brain.pkl")

print("📊 Fetching live market data for GC=F (Gold)...")
gold = yf.Ticker("GC=F")
data = gold.history(period="5d", interval="15m")

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

data.dropna(inplace=True)

latest_candle = data.iloc[-1]
features = ['Dist_to_EMA21', 'RSI_Slope', 'Volume_Spike', 'ATR', 'Hour']
live_features = pd.DataFrame([latest_candle[features]])

prediction = model.predict(live_features)[0]

print("\n" + "=" * 45)
print(f"⏱️ Live Gold Market Update | Hour: {int(latest_candle['Hour'])}:00")
print(f"💰 Current Price : ${latest_candle['Close']:.2f}")
print(f"📈 RSI Level     : {latest_candle['RSI']:.2f}")
print(f"🎯 EMA Distance  : {latest_candle['Dist_to_EMA21']:.2f}")
print("=" * 45)

if prediction == 1:
    print("🤖 AI SIGNAL: 🟢 STRONG BUY PREDICTED!")
    print(f"Take Profit : +${(latest_candle['ATR'] * 3.75):.2f} (Level: ${latest_candle['Close'] + (latest_candle['ATR'] * 3.75):.2f})")
    print(f"Stop-Loss   : -${(latest_candle['ATR'] * 1.5):.2f} (Level: ${latest_candle['Close'] - (latest_candle['ATR'] * 1.5):.2f})")
else:
    print("🤖 AI SIGNAL: 🔴 WAIT / IGNORE")
    print("Market is noisy. AI suggests staying out.")
print("=" * 45)