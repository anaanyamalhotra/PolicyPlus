import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import traceback

st.set_page_config(page_title="PolicyPulse | AI Impact Tracker", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("chr_multi_year.csv", dtype={"FIPS": str})
    return df

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    st.title("📊 PolicyPulse: AI-Powered Policy Impact Tracker")
    st.write("Explore multi-year U.S. health & economic data to model and visualize policy impacts.")

    df = load_data()

    features = ["UnemploymentRate", "UninsuredAdults", "AccessToCareIndex"]
    target = "PreventableHospitalStays"
    df = df.dropna(subset=features + [target])

    # Sidebar
    with st.sidebar:
        st.header("🔧 App Settings")
        selected_year = st.selectbox("Year", sorted(df['Year'].unique()), key="year_sidebar")
        st.markdown("---")
        st.subheader("📘 About")
        st.markdown("""
        **PolicyPulse** predicts preventable hospitalizations based on socioeconomic inputs using machine learning.
        - 🎯 Choose a scenario or input manually
        - 📉 See predictions and model performance
        - 🌎 Visualize outcomes by year and geography
        - 📤 Upload your own data for live prediction
        """)

    # Presets
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

    # Prepare data and train model
    X = df[features]
    y = np.log1p(df[target].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.subheader("📈 Model Performance")
    st.metric("Root Mean Squared Error", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    # Prediction
    input_data = np.array([[unemployment, uninsured, access_index]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_value = np.expm1(prediction[0])
    st.success(f"🧠 Predicted Preventable Hospital Stays: **{predicted_value:.0f}**")

    # Feature importance
    st.subheader("🔍 Feature Importance")
    importances = model.feature_importances_
    for feat, score in zip(features, importances):
        st.write(f"- **{feat}**: {score:.2f}")

    # Fit plot
    st.subheader("📉 Model Fit: Actual vs. Predicted")
    fig2, ax = plt.subplots()
    ax.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig2)

    # Yearly bar plot
    st.subheader("📅 Yearly Trends in Hospital Stays")
    df_year = df[df['Year'] == selected_year]
    fig3 = px.bar(df_year.sort_values("PreventableHospitalStays", ascending=False),
                  x="State", y="PreventableHospitalStays",
                  title=f"Preventable Hospital Stays in {selected_year}")
    st.plotly_chart(fig3)

    # US state name → abbreviation mapping
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
        'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    df_year["StateCode"] = df_year["State"].map(state_abbrev)

    # Choropleth map
    st.subheader("🗺️ US Map of Hospital Stays")
    fig4 = px.choropleth(
        df_year,
        locations="StateCode",
        locationmode="USA-states",
        color="PreventableHospitalStays",
        color_continuous_scale="OrRd",
        scope="usa",
        title=f"Preventable Hospital Stays by State ({selected_year})"
    )
    st.plotly_chart(fig4)

    # Upload + Predict
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

            # Download
            csv_data = user_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", data=csv_data, file_name="policy_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Something went wrong with your upload.")
            st.text(str(e))
    # GPT-style explanation
    st.subheader("🧠 Explain the Prediction (Experimental)")
    explanation_prompt = f"""
    Given UnemploymentRate={unemployment}, UninsuredAdults={uninsured}, and AccessToCareIndex={access_index},
    explain in simple language why the predicted preventable hospital stays might be {int(predicted_value)}.
    """
    st.code(explanation_prompt, language="markdown")
    st.markdown("**Simulated GPT Response:**")
    st.success("Based on high uninsured rates and limited access to care, the model predicts elevated hospitalizations due to delayed preventive treatment.")

    # Scenario Comparison Dashboard
    st.subheader("🆚 Compare Two Policy Scenarios")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scenario A**")
        ua = st.slider("Unemployment Rate (A)", 0.0, 15.0, 5.0, key="ua")
        ui = st.slider("Uninsured Adults (%) (A)", 0.0, 25.0, 10.0, key="ui")
        ac = st.slider("Access to Care Index (A)", 0.0, 10.0, 5.0, key="ac")
        input_scaled_a = scaler.transform(np.array([[ua, ui, ac]]))
        pred_a = np.expm1(model.predict(input_scaled_a)[0])

    with col2:
        st.markdown("**Scenario B**")
        ub = st.slider("Unemployment Rate (B)", 0.0, 15.0, 7.0, key="ub")
        uj = st.slider("Uninsured Adults (%) (B)", 0.0, 25.0, 14.0, key="uj")
        ad = st.slider("Access to Care Index (B)", 0.0, 10.0, 4.0, key="ad")
        input_scaled_b = scaler.transform(np.array([[ub, uj, ad]]))
        pred_b = np.expm1(model.predict(input_scaled_b)[0])

    st.write("📊 **Predicted Preventable Hospital Stays**")
    st.bar_chart(pd.DataFrame({
        "Scenario": ["A", "B"],
        "Prediction": [pred_a, pred_b]
    }).set_index("Scenario"))

    # Animated Choropleth
    st.subheader("🎞 Animated US Trends Over Time")
    df_map = df.copy()
    df_map["StateCode"] = df_map["State"].map(state_abbrev)
    fig5 = px.choropleth(
        df_map,
        locations="StateCode",
        locationmode="USA-states",
        color="PreventableHospitalStays",
        animation_frame="Year",
        color_continuous_scale="YlOrRd",
        scope="usa",
        title="Preventable Hospital Stays by State Over Time"
    )
    st.plotly_chart(fig5)

    # Feedback form
    st.subheader("💬 Feedback")
    with st.form("feedback_form"):
        feedback_text = st.text_area("Tell us what you think:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("✅ Thank you for your feedback!")

    # Feedback log
    st.subheader("📝 Save Feedback (Simulated)")
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if submitted and feedback_text:
        st.session_state.feedback_log.append(feedback_text)

    if st.session_state.feedback_log:
        with st.expander("📂 View Submitted Feedback Log"):
            for i, fb in enumerate(st.session_state.feedback_log):
                st.markdown(f"**{i+1}.** {fb}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Something went wrong.")
        st.text(traceback.format_exc())


