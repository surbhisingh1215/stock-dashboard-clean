import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Title
st.title("📈 AI-Powered Stock Market Dashboard")

# Input stock symbol
symbol = st.text_input("Enter stock symbol (e.g., AAPL):", value="AAPL")

if symbol:
    stock = yf.Ticker(symbol)
    data = stock.history(period="6mo")

    if data.empty:
        st.error("⚠️ Stock not found. Please enter a valid symbol.")
    else:
        st.subheader(f"{symbol} Stock Data")
        st.dataframe(data)

        # 📌 AI Model: Predict Next 7 Days
        data["Days"] = np.arange(len(data)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(data["Days"].values.reshape(-1, 1), data["Close"].values.reshape(-1, 1))

        future_days = np.arange(len(data) + 1, len(data) + 8).reshape(-1, 1)
        predictions = model.predict(future_days)

        prediction_df = pd.DataFrame({
        "Date": pd.date_range(start=data.index[-1], periods=8, freq="D")[1:].to_numpy().flatten(),
        "Predicted Close": predictions.flatten()
        })
        st.subheader("📌 AI-Predicted Stock Prices (Next 7 Days)")
        st.dataframe(prediction_df)

        # 📈 Interactive Chart (Actual vs Predicted)
        fig = px.line(title=f"{symbol} Actual & Predicted Prices")
        fig.add_scatter(x=data.index, y=data["Close"], mode="lines", name="Actual Price")
        fig.add_scatter(x=prediction_df["Date"], y=prediction_df["Predicted Close"], mode="lines", name="Predicted Price", line=dict(dash="dot"))
        st.plotly_chart(fig)
