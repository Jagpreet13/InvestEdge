import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt
from keras.models import load_model # type: ignore
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

Base.metadata.create_all(engine)

# Password hash and verification functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

# Main app logic
def main():
    st.title("Stock Trend Prediction")
    # Page settings
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

    elif page == 'login':
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                user = session.query(User).filter_by(username=username).first()
                if user and verify_password(user.password, password):
                    st.session_state['user'] = username
                    st.session_state['page'] = 'stock'
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

    elif page == 'stock':
        if 'user' in st.session_state:
            user_input = st.text_input('Enter stock ticker', 'AAPL')
            start = '2019-01-01'
            end = '2024-04-01'

            stock = yf.Ticker(user_input)
            df = stock.history(start=start, end=end)

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
            plt.plot(ma100, 'r')
            plt.plot(ma200, 'g')
            plt.plot(df.Close, 'b')
            st.pyplot(fig)

            # Splitting data into training and testing
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

            # Using MinMaxScaler for normalization
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
            scaler_scale = 1 / scaler.scale_[0]
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

        else:
            st.session_state['page'] = 'login'
            st.experimental_rerun()

# Run the main function
if __name__ == "__main__":
    main()
