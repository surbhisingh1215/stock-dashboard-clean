import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Title of the app
st.title("📈 AI-Powered Stock Market Dashboard")

# Input for stock symbol
symbol = st.text_input("Enter stock symbol (e.g., AAPL):", value="AAPL")

if symbol:
    # Fetch stock data using Yahoo Finance
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")  # Fetch 1 year of data

    # If stock data is not found
    if data.empty:
        st.error("⚠️ Stock not found. Please enter a valid symbol.")
    else:
        # Show stock data
        st.subheader(f"{symbol} Stock Data")
        st.dataframe(data)

        # Prepare data for LSTM model
        data_close = data[['Close']]  # We will only use the 'Close' price
        data_values = data_close.values  # Convert DataFrame to NumPy array
        data_values = data_values.astype('float32')  # Convert to float32

        # Normalize the data to fit in the range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_values = scaler.fit_transform(data_values)

        # Check if there's enough data for splitting
        if len(data_values) < 60:  # We need at least 60 data points for the time_step
            st.error("Not enough data to create training and testing datasets.")
            st.stop()

        # Split the data into train and test sets
        train_size = int(len(data_values) * 0.8)
        train_data = data_values[:train_size]
        test_data = data_values[train_size:]

        # Function to create dataset for LSTM
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), 0])  # X will contain sequences
                y.append(dataset[i + time_step, 0])      # y will contain next day's close
            return np.array(X), np.array(y)

        # Time step for LSTM
        time_step = 30  # Reduced time_step to 30
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Ensure proper reshaping for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))  # Output layer with 1 unit (the predicted close price)
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # **Fix: Get the dates from the original data for plotting**
        last_date = data.index[-1]  # Get the last date from the DataFrame index

        # Generate future dates based on predicted stock prices length
        future_dates = pd.date_range(last_date, periods=len(predicted_stock_price) + 1, freq='D')[1:]  # Generate next N days

        # Create DataFrame for predictions
        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": predicted_stock_price.flatten()
        })

        # Show predicted stock prices for the next N days
        st.subheader(f"AI-Predicted Stock Prices for Next {len(predicted_stock_price)} Days")
        st.dataframe(prediction_df)

        # Plot the actual vs predicted stock prices
        fig = px.line(title=f"{symbol} Actual & Predicted Prices")
        fig.add_scatter(x=data.index[:len(predicted_stock_price)], y=data_close['Close'][:len(predicted_stock_price)], mode="lines", name="Actual Price")
        fig.add_scatter(x=prediction_df["Date"], y=prediction_df["Predicted Close"], mode="lines", name="Predicted Price", line=dict(dash="dot"))
        st.plotly_chart(fig)
