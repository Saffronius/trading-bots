import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
from math import sqrt

def fetch_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "max", "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["Timestamp", "Price"])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index("Timestamp", inplace=True)
    return df

# Fetch and prepare data
data = fetch_btc_data()

# Calculate Technical Indicators
data['MACD'] = ta.trend.macd(data['Price'])
data['RSI'] = ta.momentum.rsi(data['Price'])

# Keep only MACD and RSI
data = data[['MACD', 'RSI']]

# Drop NaN values
data.dropna(inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

def create_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])
        # Y needs to be defined. In this case, we'll predict the next day's MACD value as an example
        # Adjust accordingly if you're predicting something different
        Y.append(data[i, 0]) # Assuming the next day's MACD is the target
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Adjust the model as per the new shape
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=32)

predicted = model.predict(X_test)

# Since we're predicting MACD (or any other single feature), we can directly inverse transform
predicted_inversed = scaler.inverse_transform(np.concatenate((predicted, np.zeros((predicted.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
y_test_inversed = scaler.inverse_transform(np.concatenate((y_test.reshape(-1,1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

def apply_stop_loss_and_take_profit(actual_prices, predicted_prices, stop_loss_pct=0.1, take_profit_pct=0.2):
    positions = np.zeros(len(actual_prices))
    entry_price = 0
    for i in range(1, len(actual_prices)):
        if positions[i-1] == 0 and predicted_prices[i] > actual_prices[i-1]:
            positions[i] = 1  # Enter long position
            entry_price = actual_prices[i-1]
        elif positions[i-1] == 1:
            if actual_prices[i] < entry_price * (1 - stop_loss_pct):
                positions[i] = 0  # Exit long position due to stop-loss
            elif actual_prices[i] > entry_price * (1 + take_profit_pct):
                positions[i] = 0  # Exit long position due to take-profit
            elif predicted_prices[i] < actual_prices[i]:
                positions[i] = 0  # Exit long position based on prediction
            else:
                positions[i] = 1  # Maintain long position
    return positions

# Get the actual prices for the test period
actual_prices = data['Price'][len(data) - len(y_test_inversed):]

# Apply stop-loss and take-profit to the predicted prices
positions = apply_stop_loss_and_take_profit(actual_prices, predicted_inversed)

# Calculate and print RMSE
rmse = sqrt(mean_squared_error(y_test_inversed, predicted_inversed))
print(f"Root Mean Squared Error: {rmse}")

# Calculate and print RMSE
rmse = sqrt(mean_squared_error(y_test_inversed, predicted_inversed))
print(f"Root Mean Squared Error: {rmse}")
