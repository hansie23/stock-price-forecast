import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objects as go

# Load the model and scaler
@st.cache_resource
def load_lstm_model():
    model = pickle.load(open("stocks_lstm.pkl", "rb"))
    return model

@st.cache_data
def load_scaler():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return scaler

model = load_lstm_model()
scaler = load_scaler()

# Function to forecast future values
def forecast_future(model, scaler, last_sequence, future_days):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        # predict the next value
        next_prediction = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(next_prediction[0, 0])

        # update the sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_prediction[0, 0]

    # inverse transform the predictions to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# load dummy dataset (BRITANNIA.csv)
def load_dummy_data():
    dummy_data = pd.read_csv("BRITANNIA.csv", index_col="Date", parse_dates=True)
    return dummy_data

# Streamlit app
st.title("Stock Price Forecasting using LSTM")

# option to upload a file or use the dummy dataset
st.sidebar.subheader("Choose Dataset")
use_dummy_data = st.sidebar.checkbox("Use Dummy Dataset (BRITANNIA.csv)", value=True)

if use_dummy_data:
    data = load_dummy_data()
    st.sidebar.write("Using the dummy dataset: BRITANNIA.csv")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your stock data (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col="Date", parse_dates=True)
    else:
        st.write("Please upload a CSV file or use the dummy dataset to get started.")
        st.stop()

# keep only the 'Open' column
data = data[["Open"]]

# display the data
st.subheader("Stock Data")
st.write(data)

# preprocess the data
scaled_data = scaler.transform(data)

# prepare the data for prediction
X_test = []
for i in range(50, len(scaled_data)):
    X_test.append(scaled_data[i - 50 : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# make predictions
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)

# Create a DataFrame for visualization
valid = data[50:]
valid["Predictions"] = pred

# plot the results
st.subheader("Actual vs Predicted Stock Prices")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=valid.index, y=valid["Open"], mode="lines", name="Actual Prices"))
fig1.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"], mode="lines", name="Predicted Prices", line=dict(color="red")))
fig1.update_layout(title="Actual vs Predicted Stock Prices", xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig1)

# display evaluation metrics
st.subheader("Evaluation Metrics")
mse = np.mean(np.square(valid["Open"] - valid["Predictions"]))
rmse = np.sqrt(mse)
mape = np.mean(np.abs((valid["Open"] - valid["Predictions"]) / valid["Open"])) * 100

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Future forecasting
st.subheader("Future Forecasting")
future_days = st.slider("Select the number of days to forecast (up to 30 days):", 1, 30, 10)

if st.button("Forecast Future Prices"):
    # Get the last sequence of 50 days for forecasting
    last_sequence = scaled_data[-50:]

    # Forecast future values
    future_predictions = forecast_future(model, scaler, last_sequence, future_days)

    # Create a DataFrame for future predictions
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame(index=future_dates, data=future_predictions, columns=["Forecasted Prices"])

    # Combine historical data with future predictions
    historical_data = data[-90:]        # Last 3 months of data
    combined_data = pd.concat([historical_data, future_df], axis=0)

    # Plot the combined data
    st.subheader(f"Forecasted Stock Prices for the Next {future_days} Days (Connected with Last 3 Months)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=historical_data.index, y=historical_data["Open"], mode="lines", name="Historical Prices (Last 3 Months)"))
    fig2.add_trace(go.Scatter(x=future_df.index, y=future_df["Forecasted Prices"], mode="lines", name="Forecasted Prices", line=dict(color="red")))
    fig2.update_layout(title=f"Forecasted Stock Prices for the Next {future_days} Days", xaxis_title="Date", yaxis_title="Stock Price")
    st.plotly_chart(fig2)

    # Display the forecasted values
    st.write(future_df)
