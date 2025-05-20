# Content omitted fo# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import traceback

# Streamlit page config
st.set_page_config(page_title="PolicyPulse | AI Impact Tracker", layout="centered")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("chr_multi_year.csv", dtype={"FIPS": str})
    df = df[df["Level"] == "State"]
    return df
print(df.columns)
# Train ML model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Main app logic
def main():
    st.title("📊 PolicyPulse: AI-Powered Policy Impact Tracker")
    st.write("Explore multi-year U.S. health & economic data to model and visualize policy impacts.")

    df = load_data()

    features = ["UnemploymentRate", "UninsuredAdults", "AccessToCareIndex"]
    target = "PreventableHospitalStays"

    if not all(col in df.columns for col in features + [target]):
        st.error("One or more required columns are missing in the dataset.")
        return

    # Prepare data
    X = df[features]
    y = np.log1p(df[target].values)  # log-transform for stability

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display performance
    st.subheader("📈 Model Performance")
    st.metric("Root Mean Squared Error", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    # Simulate a policy scenario
    st.subheader("📌 Simulate a Policy Scenario")
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 15.0, 5.0)
    uninsured = st.slider("Uninsured Adults (%)", 0.0, 25.0, 10.0)
    access_index = st.slider("Access to Care Index (0 = poor, 10 = great)", 0.0, 10.0, 5.0)

    input_data = np.array([[unemployment, uninsured, access_index]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_value = np.expm1(prediction[0])  # Reverse log1p transform

    st.success(f"🧠 Predicted Preventable Hospital Stays: **{predicted_value:.0f}**")

    # Visualization
    st.subheader("📊 Data Visualization")
    fig = px.scatter(
        df,
        x="UnemploymentRate",
        y="PreventableHospitalStays",
        color="State",
        title="Unemployment Rate vs. Preventable Hospital Stays"
    )
    st.plotly_chart(fig)

# Run app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Something went wrong.")
        st.text(traceback.format_exc())
