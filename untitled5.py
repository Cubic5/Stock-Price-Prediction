# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/108mOW0xIJ8_685bMtry0a-8POx6JOmQX
"""


# Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import seaborn as sns
import yfinance as yf
import joblib, os

# Loading the saved model
saved_model = joblib.load('arima_model.pkl')

# Define a function to fetch the stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Define a function to apply Box-Cox transformation
def apply_boxcox_transformation(data):
    df_arima = data[['Close']] # Extract the 'Close' price column
    df_arima['Close'], _ = boxcox(df_arima['Close'])
    return df_arima

# Define a function to make predictions
def predict_stock_price(model, data):
    prediction = model.predict(data)
    return prediction

# Streamlit App
st.title('Stock Price Prediction App')

# Get user input
ticker = st.text_input('Enter the stock ticker symbol (e.g. AAPL)')
start_date = st.date_input('Select the start date')
end_date = st.date_input('Select the end date')

# Validate dates
if start_date > end_date:
    st.error('Error: Start date must be before end date.')
else:
    # Initialize session state for transformed_data
    if 'transformed_data' not in st.session_state:
        st.session_state.transformed_data = None

    # Fetch and display the stock data
    if st.button('Get Data'):
        data = fetch_stock_data(ticker, start_date, end_date)


    if data.empty:
        st.write('No data found for the select date range and ticker.')
    else:
        st.write('Stock Data (Original):')
        st.write(data)

        # Apply the Box-Cox transformation
        st.session_state.transformed_data = apply_boxcox_transformation(data)
        st.write('Stock Data (After Box-Cox Transformation):')
        st.write(st.session_state.transformed_data)


# Only enable Predictions if transformed_data is available
if st.button('Predict'):
    if st.session_state.transformed_data is not None:
        # Make predictions on the transformed data
        prediction = predict_stock_price(saved_model, st.session_state.transformed_data)
        st.write('Predicted Stock Prices (Ater Transformation):')
        st.write(prediction)
    else:
        st.write('Please etch the data first.')


# Required to let Streamlit instantiate our web app.
if __name__ == '__main_':
        main()
