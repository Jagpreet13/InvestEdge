import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, LSTM # type: ignore

def train_and_save_model(ticker):
    # Fetching stock data
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")

    # Preprocessing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    # Creating training data
    x_train, y_train = [], []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(120, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    # Save model and scaler
    model.save('stock_model.h5')
    pd.to_pickle(scaler, 'scaler.pkl')

if __name__ == "__main__":
    train_and_save_model('AAPL')  # Train model for Apple as an example
