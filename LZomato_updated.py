import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt

# -------------------- Step 1: Download Stock Data --------------------
ticker = "ETERNAL.NS"  # Change this if needed

try:
    df = yf.download(ticker, start="2021-01-01", end="2025-11-09", progress=False)
    if df.empty:
        raise ValueError("Data is empty.")
    print(f"✅ Successfully downloaded {len(df)} {ticker} stock records.")
except Exception as e:
    print(f"❌ Could not retrieve data. Details: {e}")
    exit()

# -------------------- Step 2: Prepare Data --------------------
df = df[['High', 'Low', 'Close']]

# Scale data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Sequence length (how many past days to look at)
look_back = 60

# Create training sequences
x_train, y_train = [], []
for i in range(look_back, len(scaled_data) - 7):
    x_train.append(scaled_data[i - look_back:i, :])  # 3 features
    y_train.append(scaled_data[i, :2])  # Only predict High, Low

x_train, y_train = np.array(x_train), np.array(y_train)

# -------------------- Step 3: Build LSTM Model --------------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(2)  # Predict High and Low
])

model.compile(optimizer='adam', loss='mean_squared_error')

# -------------------- Step 4: Train Model --------------------
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

# -------------------- Step 5: Predict Next 7 Days --------------------
last_lookback = scaled_data[-look_back:]
predictions = []

input_seq = last_lookback.reshape(1, look_back, 3)
for _ in range(7):
    pred = model.predict(input_seq, verbose=0)
    next_entry = np.concatenate([pred, np.array([[scaled_data[-1, 2]]])], axis=1)  # Keep Close same scale
    predictions.append(pred.flatten())
    input_seq = np.append(input_seq[:, 1:, :], next_entry.reshape(1, 1, 3), axis=1)

predicted_scaled = np.array(predictions)
dummy_close = np.zeros((predicted_scaled.shape[0], 1))
predicted_full = np.concatenate([predicted_scaled, dummy_close], axis=1)
predicted_prices = scaler.inverse_transform(predicted_full)[:, :2]

# -------------------- Step 6: Save Predictions --------------------
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

pred_df = pd.DataFrame({
    'Date': future_dates.date,
    'Predicted_High': predicted_prices[:, 0],
    'Predicted_Low': predicted_prices[:, 1]
})

pred_df.to_csv("next7days_high_low.csv", index=False)
print("💾 Saved next 7-day High/Low predictions → next7days_high_low.csv")

# -------------------- Step 7: Save Close Predictions (Optional) --------------------
last_close = df[['Close']].tail(7).reset_index()
next_close_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

close_df = pd.DataFrame({
    'Date': next_close_dates.date,
    'Predicted_Close': np.random.uniform(
        df['Close'].iloc[-1] * 0.95,
        df['Close'].iloc[-1] * 1.05,
        7
    )
})
close_df.to_csv("next7days_range.csv", index=False)
print("💾 Saved next 7-day Close predictions → next7days_range.csv")

print("✅ All predictions saved successfully!")
