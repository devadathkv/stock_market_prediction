import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:/stock_market_prediction/archive/NIFTY50_all.csv")
    return df

# Function to preprocess data and create sequences
def preprocess_data(df, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i])
        
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Function to train or load model
def get_trained_model(ticker, X_train, y_train):
    model_path = f"models/{ticker}_model.h5"
    if os.path.exists(model_path):
        st.success("✅ Loaded pre-trained model")
        model = load_model(model_path)
    else:
        st.warning("⚠️ No pre-trained model found. Training a new one...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        st.success("✅ Model trained and saved!")
    return model

# Load data
data = load_data()
tickers = sorted(data['Symbol'].unique())

# Streamlit UI
st.title("📈 Stock Price Prediction")
st.markdown("Select a stock and predict future prices with LSTM")

# Stock selector
selected_ticker = st.selectbox("Select a Stock", tickers)

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(data[data['Symbol'] == selected_ticker].head())

# Predict button
if st.button("🔮 Predict"):
    st.write(f"### 📊 Prediction for {selected_ticker}")

    # Filter selected stock
    stock_df = data[data['Symbol'] == selected_ticker][['Date', 'Close']].copy()
    stock_df = stock_df.sort_values('Date')
    stock_df.reset_index(drop=True, inplace=True)

    # Preprocess data
    time_step = 60
    X, y, scaler = preprocess_data(stock_df, time_step)
    split = int(len(X)*0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Train or load model
    model = get_trained_model(selected_ticker, X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Inverse transform predictions
    train_pred = scaler.inverse_transform(train_pred)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_pred = scaler.inverse_transform(test_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(stock_df['Close'], label="Actual Price", color="blue")
    plt.plot(range(time_step, split+time_step), train_pred, label="Train Prediction", color="orange")
    plt.plot(range(split+time_step, len(stock_df)), test_pred, label="Test Prediction", color="green")
    plt.title(f"{selected_ticker} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
