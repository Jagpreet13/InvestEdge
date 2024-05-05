import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import bcrypt
from keras.models import load_model # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///users.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)
session = Session()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    watchlist = relationship("Watchlist", back_populates="user")

class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    stock_symbol = Column(String, nullable=False)
    user = relationship("User", back_populates="watchlist")

# Password hash and verification functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

# Load the predictive model
model = load_model('stock_model.h5')

# Function to calculate signals based on predicted price and recent trends
def generate_signal(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(start='2019-01-01', end='2024-04-01')

    # Calculate the moving average (100-day)
    df['MA'] = df['Close'].rolling(window=100).mean()

    # Prepare data for the model
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])

    x_test = np.array(x_test)
    predicted_prices = model.predict(x_test)

    # Convert predictions back to the original scale
    scaler_scale = 1 / scaler.scale_[0]
    predicted_prices = predicted_prices * scaler_scale

    # Get the most recent predicted price, moving average price, and latest actual price
    latest_predicted_price = predicted_prices[-1][0]
    latest_actual_price = df['Close'].iloc[-1]
    latest_avg_price = df['MA'].iloc[-1]

    # Determine the recent trend using a linear regression slope
    recent_days = 10
    recent_prices = df['Close'][-recent_days:].values
    reg = LinearRegression().fit(np.arange(recent_days).reshape(-1, 1), recent_prices)
    trend_slope = reg.coef_[0]
    upward_trend = trend_slope > 0

    # Generate trading signals
    if latest_predicted_price < latest_avg_price:
        if upward_trend:
            return "Buy", latest_predicted_price, latest_avg_price, latest_actual_price
        else:
            return "Sell", latest_predicted_price, latest_avg_price, latest_actual_price
    else:
        return "Hold", latest_predicted_price, latest_avg_price, latest_actual_price

# Main application logic
def main():
    st.title("Stock Trend Prediction")
    page = st.session_state.get('page', 'signup')

    if page == 'signup':
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")

            if submit:
                hashed_password = hash_password(password)
                user = User(username=username, password=hashed_password.decode('utf-8'))
                session.add(user)
                session.commit()
                st.success("You have successfully signed up! Please log in.")
                st.session_state['page'] = 'login'
                st.experimental_rerun()

        if st.button("Already have an account? Log in here!"):
            st.session_state['page'] = 'login'
            st.experimental_rerun()

    elif page == 'login':
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                user = session.query(User).filter_by(username=username).first()
                if user and verify_password(user.password, password):
                    st.session_state['user'] = username
                    st.session_state['page'] = 'dashboard'
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

    elif page == 'dashboard':
        if 'user' in st.session_state:
            username = st.session_state['user']
            user = session.query(User).filter_by(username=username).first()

            st.subheader(f"Welcome, {username}")
            st.write("---")
            st.subheader("Manage your stock watchlist")
            stock_symbol = st.text_input("Enter stock ticker to add to watchlist")
            add_button = st.button("Add to Watchlist")

            if add_button and stock_symbol:
                if len(user.watchlist) < 5:
                    new_entry = Watchlist(user_id=user.id, stock_symbol=stock_symbol)
                    session.add(new_entry)
                    session.commit()
                else:
                    st.warning("You can only add up to 5 stocks in your watchlist.")

            st.subheader("Your Watchlist")
            if user.watchlist:
                columns = st.columns(min(len(user.watchlist), 5))
                for idx, stock in enumerate(user.watchlist):
                    stock_info = yf.Ticker(stock.stock_symbol).info
                    signal, predicted_price, avg_price, actual_price = generate_signal(stock.stock_symbol)
                    with columns[idx]:
                        st.write(f"**{stock.stock_symbol}**")
                        st.write(f"Name: {stock_info.get('shortName', 'N/A')}")
                        st.write(f"Original Price: {actual_price:.2f}")
                        st.write(f"Predicted Price: {predicted_price:.2f}")
                        st.write(f"Average Price: {avg_price:.2f}")
                        st.write(f"Signal: {signal}")

                        # Visual bar showing the current performance
                        performance = min(predicted_price / avg_price, 1.0) if avg_price else 0.0
                        st.progress(performance)

                        if st.button(f"Predict {stock.stock_symbol}"):
                            st.session_state['stock_symbol'] = stock.stock_symbol
                            st.session_state['page'] = 'stock'
                            st.experimental_rerun()
                        if st.button(f"Remove {stock.stock_symbol}"):
                            session.query(Watchlist).filter_by(id=stock.id).delete()
                            session.commit()
                            st.experimental_rerun()  # Refresh the page to show updated watchlist
            else:
                st.write("No stocks in the watchlist.")

    elif page == 'stock':
        if 'stock_symbol' in st.session_state:
            user_input = st.session_state['stock_symbol']
            start = '2019-01-01'
            end = '2024-04-01'

            stock = yf.Ticker(user_input)
            df = stock.history(start=start, end=end)

            st.subheader(f'Data from 2019-2024 for {user_input}')
            st.write(df.describe())

            st.subheader('Closing Price vs Time chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time chart with 100MA')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100)
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time chart with 100MA & 200MA')
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, 'r')
            plt.plot(ma200, 'g')
            plt.plot(df.Close, 'b')
            st.pyplot(fig)

            data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            model = load_model('stock_model.h5')

            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.transform(final_df)

            x_test, y_test = [], []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100: i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler_scale = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scaler_scale
            y_test = y_test * scaler_scale

            st.subheader('Predictions vs Original')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)

            if st.button("Back to Dashboard"):
                st.session_state['page'] = 'dashboard'
                st.experimental_rerun()

# Run the main function
if __name__ == "__main__":
    main()
