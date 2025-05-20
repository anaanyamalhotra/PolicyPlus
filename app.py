import matplotlib.pyplot as plt

# --- Policy Presets ---
policy_options = {
    "Manual Input": None,
    "Status Quo": (5.0, 10.0, 5.0),
    "Medicaid Expansion": (4.0, 6.5, 8.5),
    "Benefit Cuts": (6.5, 14.0, 3.5)
}
policy_choice = st.selectbox("Choose a Policy Scenario", options=policy_options.keys())
if policy_choice != "Manual Input":
    unemployment, uninsured, access_index = policy_options[policy_choice]
    st.success(f"Preset selected: {policy_choice}")
else:
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 15.0, 5.0)
    uninsured = st.slider("Uninsured Adults (%)", 0.0, 25.0, 10.0)
    access_index = st.slider("Access to Care Index", 0.0, 10.0, 5.0)

# --- Model prediction for inputs ---
input_data = np.array([[unemployment, uninsured, access_index]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_value = np.expm1(prediction[0])  # reverse log1p
st.success(f"🧠 Predicted Preventable Hospital Stays: **{predicted_value:.0f}**")

# --- Feature Importance ---
st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
for feat, score in zip(features, importances):
    st.write(f"- **{feat}**: {score:.2f}")

# --- Actual vs Predicted Plot ---
st.subheader("📉 Model Fit: Actual vs. Predicted")
fig2, ax = plt.subplots()
ax.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.6)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig2)

# --- Trend by Year ---
st.subheader("📅 Yearly Trends in Hospital Stays")
selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
df_year = df[df['Year'] == selected_year]
fig3 = px.bar(df_year.sort_values("PreventableHospitalStays", ascending=False),
              x="State", y="PreventableHospitalStays",
              title=f"Preventable Hospital Stays in {selected_year}")
st.plotly_chart(fig3)

# --- Upload Custom Data ---
st.subheader("📤 Upload Your Own Data")
user_csv = st.file_uploader("Upload a CSV with columns: UnemploymentRate, UninsuredAdults, AccessToCareIndex", type="csv")
if user_csv is not None:
    try:
        user_df = pd.read_csv(user_csv)
        user_input_scaled = scaler.transform(user_df[features])
        user_preds = model.predict(user_input_scaled)
        user_df["PredictedPreventableHospitalStays"] = np.expm1(user_preds)
        st.write("✅ Predictions for your data:")
        st.dataframe(user_df)
    except Exception as e:
        st.error("Something went wrong with your upload.")
        st.text(str(e))

