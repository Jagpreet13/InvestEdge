import yfinance as yf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import load_model # type: ignore
import streamlit as st 

# Streamlit interface setup
st.title('Stock Trend Prediction') 

# User input for stock ticker
user_input = st.text_input('Enter stock ticker', 'AAPL')
start = '2019-01-01'
end = '2024-04-01'

# Fetching stock data using yfinance
stock = yf.Ticker(user_input)
df = stock.history(start=start, end=end)

# Displaying stock data
st.subheader(f'Data from 2019-2024 for {user_input}')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart') 
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r') 
plt.plot(ma200 , 'g')
plt.plot(df.Close , 'b')
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Using MinMaxScaler for normalization
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load model and scaler
model = load_model('stock_model.h5')


# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler_scale = 1 / scaler.scale_[0]  # adjusted this part to avoid overriding `scaler`
y_predicted = y_predicted * scaler_scale
y_test = y_test * scaler_scale

# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)