import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
from fpdf import FPDF
from datetime import datetime

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
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
    return model

# Function to create PDF report
def create_pdf_report(ticker1, ticker2, comparison_plot_path, stats_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Title
    pdf.cell(0, 10, f"Stock Comparison Report: {ticker1} vs {ticker2}", 0, 1, 'C')
    pdf.ln(10)
    
    # Date
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(10)
    
    # Add comparison plot
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Price Comparison", 0, 1)
    pdf.image(comparison_plot_path, x=10, w=180)
    pdf.ln(10)
    
    # Add statistics table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Performance Statistics", 0, 1)
    pdf.ln(5)
    
    # Create table
    col_width = pdf.w / 3
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_width, 10, "Metric", 1)
    pdf.cell(col_width, 10, ticker1, 1)
    pdf.cell(col_width, 10, ticker2, 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 12)
    for index, row in stats_df.iterrows():
        pdf.cell(col_width, 10, index, 1)
        pdf.cell(col_width, 10, str(round(row[ticker1], 2)), 1)
        pdf.cell(col_width, 10, str(round(row[ticker2], 2)), 1)
        pdf.ln()
    
    # Save the PDF
    report_path = f"reports/{ticker1}_vs_{ticker2}_report.pdf"
    os.makedirs("reports", exist_ok=True)
    pdf.output(report_path)
    return report_path

# Load data
data = load_data()
tickers = sorted(data['Symbol'].unique())

# Streamlit UI
st.title("📈 Stock Price Prediction & Comparison")
st.markdown("Compare two stocks and predict future prices with LSTM")

# Stock selectors
col1, col2 = st.columns(2)
with col1:
    selected_ticker1 = st.selectbox("Select First Stock", tickers, index=tickers.index('RELIANCE') if 'RELIANCE' in tickers else 0)
with col2:
    selected_ticker2 = st.selectbox("Select Second Stock", tickers, index=tickers.index('TCS') if 'TCS' in tickers else 1)

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(f"### {selected_ticker1} Data")
    st.dataframe(data[data['Symbol'] == selected_ticker1].head())
    st.write(f"### {selected_ticker2} Data")
    st.dataframe(data[data['Symbol'] == selected_ticker2].head())

# Predict and compare button
if st.button("🔮 Predict & Compare"):
    st.write(f"### 📊 Comparison: {selected_ticker1} vs {selected_ticker2}")
    
    # Create tabs for each stock
    tab1, tab2, tab3 = st.tabs([selected_ticker1, selected_ticker2, "Comparison"])
    
    # Process first stock
    with tab1:
        stock1_df = data[data['Symbol'] == selected_ticker1][['Date', 'Close']].copy()
        stock1_df = stock1_df.sort_values('Date')
        stock1_df.reset_index(drop=True, inplace=True)
        
        time_step = 60
        X1, y1, scaler1 = preprocess_data(stock1_df, time_step)
        split = int(len(X1)*0.8)
        X1_train, y1_train = X1[:split], y1[:split]
        X1_test, y1_test = X1[split:], y1[split:]
        
        model1 = get_trained_model(selected_ticker1, X1_train, y1_train)
        st.success(f"✅ {selected_ticker1} model loaded/trained")
        
        test_pred1 = model1.predict(X1_test)
        test_pred1 = scaler1.inverse_transform(test_pred1)
        y1_test_inv = scaler1.inverse_transform(y1_test.reshape(-1, 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(stock1_df['Close'], label="Actual Price", color="blue")
        plt.plot(range(split+time_step, len(stock1_df)), test_pred1, label="Test Prediction", color="green")
        plt.title(f"{selected_ticker1} Stock Price Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
    
    # Process second stock
    with tab2:
        stock2_df = data[data['Symbol'] == selected_ticker2][['Date', 'Close']].copy()
        stock2_df = stock2_df.sort_values('Date')
        stock2_df.reset_index(drop=True, inplace=True)
        
        X2, y2, scaler2 = preprocess_data(stock2_df, time_step)
        split = int(len(X2)*0.8)
        X2_train, y2_train = X2[:split], y2[:split]
        X2_test, y2_test = X2[split:], y2[split:]
        
        model2 = get_trained_model(selected_ticker2, X2_train, y2_train)
        st.success(f"✅ {selected_ticker2} model loaded/trained")
        
        test_pred2 = model2.predict(X2_test)
        test_pred2 = scaler2.inverse_transform(test_pred2)
        y2_test_inv = scaler2.inverse_transform(y2_test.reshape(-1, 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(stock2_df['Close'], label="Actual Price", color="blue")
        plt.plot(range(split+time_step, len(stock2_df)), test_pred2, label="Test Prediction", color="green")
        plt.title(f"{selected_ticker2} Stock Price Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
    
    # Comparison tab
    with tab3:
        # Normalize prices for comparison
        norm_stock1 = stock1_df['Close'] / stock1_df['Close'].iloc[0]
        norm_stock2 = stock2_df['Close'] / stock2_df['Close'].iloc[0]
        
        # Calculate performance statistics
        stats = {
            'Start Price': [stock1_df['Close'].iloc[0], stock2_df['Close'].iloc[0]],
            'End Price': [stock1_df['Close'].iloc[-1], stock2_df['Close'].iloc[-1]],
            'Total Return (%)': [
                (stock1_df['Close'].iloc[-1] - stock1_df['Close'].iloc[0]) / stock1_df['Close'].iloc[0] * 100,
                (stock2_df['Close'].iloc[-1] - stock2_df['Close'].iloc[0]) / stock2_df['Close'].iloc[0] * 100
            ],
            'Max Drawdown (%)': [
                (stock1_df['Close'].max() - stock1_df['Close'].min()) / stock1_df['Close'].max() * 100,
                (stock2_df['Close'].max() - stock2_df['Close'].min()) / stock2_df['Close'].max() * 100
            ],
            'Volatility': [
                stock1_df['Close'].pct_change().std() * np.sqrt(252),
                stock2_df['Close'].pct_change().std() * np.sqrt(252)
            ]
        }
        stats_df = pd.DataFrame(stats, index=[selected_ticker1, selected_ticker2]).T
        
        # Display comparison plot
        plt.figure(figsize=(12, 6))
        plt.plot(stock1_df['Date'], norm_stock1, label=selected_ticker1, color='blue')
        plt.plot(stock2_df['Date'], norm_stock2, label=selected_ticker2, color='orange')
        plt.title(f"Normalized Price Comparison: {selected_ticker1} vs {selected_ticker2}")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (Base=1)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot for PDF
        comparison_plot_path = f"plots/{selected_ticker1}_vs_{selected_ticker2}_comparison.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(comparison_plot_path)
        st.pyplot(plt)
        
        # Display statistics
        st.write("### Performance Comparison")
        st.dataframe(stats_df.style.format("{:.2f}"))
        
        # Create and offer PDF download
        report_path = create_pdf_report(selected_ticker1, selected_ticker2, comparison_plot_path, stats_df)
        
        with open(report_path, "rb") as file:
            st.download_button(
                label="📄 Download Comparison Report (PDF)",
                data=file,
                file_name=f"{selected_ticker1}_vs_{selected_ticker2}_report.pdf",
                mime="application/pdf"
            )