# Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import seaborn as sns
import yfinance as yf
import joblib, os
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Loading the saved model
#saved_model = joblib.load('boxcox_arima_model.pkl')

# Define a function to fetch the stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError('No data available for the selected date range and ticker.')
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Define a function to apply Box-Cox transformation
def apply_boxcox_transformation(data):
    data = data.reset_index()
    data['Date'] = data['Date'].dt.tz_localize(None)
    # data = set_index('Date')
    df_arima = data[['Date','Close']].copy() # Extract the 'Close' price column
    df_arima['Close'], boxcox_lambda = boxcox(df_arima['Close'])
    #df_arima['Date'] = df_arima['Date'].dt.tz_localize(None)
    st.session_state.boxcox_lambda = boxcox_lambda
    
    return df_arima

def train_auto_arima_model(data):
    model = auto_arima(
        data['Close'],
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        seasonal=False,
        trace=True,
        error_action='ignore',  
        suppress_warnings=True,
        stepwise=True
    )
    return model

# Define a function to make predictions
def predict_stock_price(model, data):

    transformed_data = data[['Close']].reset_index(drop=True)

    try:
        # Generate a date range to make predictions
        forecast, conf_int = model.predict(n_periods=len(transformed_data), return_conf_int=True)


        # Generate a date range starting from the last date in the original data
        last_date = data['Date'].iloc[-1]
        prediction_dates = pd.date_range(start=last_date, periods=len(forecast) + 1, freq='B')[1:]
        
        # Apply inverse Box-Cox transformation to the prediction
        inverse_predicted_close = inv_boxcox(forecast, st.session_state.boxcox_lambda)
        
        
        # Combine the dates with the predictions
        predicted_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Close': inverse_predicted_close
        })
        return predicted_df
    except Exception as e:
        raise ValueError(f'Error in prediction: {e}')

# Streamlit App
st.title('Stock Price Prediction App')

# Get user input
ticker = st.text_input('Enter the stock ticker symbol (e.g. AAPL)')
start_date = st.date_input('Select the start date')
end_date = st.date_input('Select the end date')


# Validate dates and process data only if dates are valid
if start_date > end_date:
    st.error('Error: Start date must be before end date.')
else:
    # Fetch new stock data, apply transform and train a model when 'Get Data' is clicked
    if st.button('Get Data'):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.write('No data found for the selected date range or ticker.')
        else:
            st.write('Stock Data (Original):')
            st.write(data)

            # Apply the Box-Cox transformation for the current ticker
            st.session_state.transformed_data = apply_boxcox_transformation(data)
            st.write('Stock Data (After Box-Cox Transformation):')
            st.write(st.session_state.transformed_data)

            # Train a new model and store it in session state
            st.session_state.model = train_arima_model(st.session_state.transformed_data)

  
    # Only enable Predictions if transformed_data is available
    if st.button('Predict'):
        if 'transformed_data' in st.session_state and 'model' in st.session_state:
            try:
                # Make predictions on the transformed data
                prediction_df = predict_stock_price(st.session_state.model, st.session_state.transformed_data)
                st.write('Predicted Stock Prices :')
                st.write(prediction_df)
            except Exception as e:
                st.error(f'Prediction error: {e}')
        else:
            st.write('Please fetch the data first')


# Required to let Streamlit instantiate our web app.
if __name__ == '__main_':
        main()
