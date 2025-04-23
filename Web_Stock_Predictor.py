import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Load stock data
google_data = yf.download(stock, start, end)

# Load the model
model = load_model("Latest_Stock_Price_Gru_Model.keras")

st.subheader("Stock Data")
st.write(google_data)

# Fix the splitting and test data creation
splitting_len = int(len(google_data)*0.7)
x_test = google_data[splitting_len:].copy()

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving averages plots
st.subheader('Original Close Price and MA for 250 days')
google_data["MA_for_250_days"] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data["MA_for_200_days"] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data["MA_for_100_days"] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data["MA_for_250_days"]))

# Scaling and prediction
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(google_data[['Close']].values.reshape(-1, 1))

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Current predictions
predictions = model.predict(x_data)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        'original_data': inv_y_test.reshape(-1),
        'predictions': inv_predictions.reshape(-1)
    },
    index=google_data.index[100:]
)

st.subheader("Original Values vs Predicted Values")
st.write(plotting_data)

# Future predictions
future_days = st.number_input("Enter the no of days to predict", value=10, min_value=1)
last_100_days = scaled_data[-100:]

future_predictions = []
current_prediction = last_100_days

for _ in range(future_days):
    # Reshape the data for prediction
    current_prediction_reshaped = current_prediction.reshape((1, 100, 1))
    
    # Get the prediction
    next_pred = model.predict(current_prediction_reshaped)
    
    # Append to our predictions
    future_predictions.append(next_pred[0])
    
    # Update the prediction array
    current_prediction = np.append(current_prediction[1:], next_pred)

# Convert future predictions to actual prices
future_predictions = np.array(future_predictions)
future_pred_prices = scaler.inverse_transform(future_predictions)

# Create future dates
last_date = google_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='B')

future_predictions_df = pd.DataFrame(
    future_pred_prices,
    index=future_dates,
    columns=['Predicted_Price']
)

st.subheader("Future Price Predictions")
st.write(future_predictions_df)

# Plot all predictions
st.subheader('Historical and Future Price Predictions')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data.index, google_data.Close, 'b', label='Historical Data')
plt.plot(plotting_data.index, plotting_data.predictions, 'g', label='Historical Predictions')
plt.plot(future_predictions_df.index, future_predictions_df.Predicted_Price, 'r', label='Future Predictions')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)