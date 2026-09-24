import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("fraud_detection_pipeline.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title">🛡️ Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Transaction Fraud Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Transaction Details
# -----------------------------

st.subheader("💳 Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Balance Before",
        min_value=0.0,
        value=10000.0
    )

    newbalanceOrig = st.number_input(
        "Sender Balance After",
        min_value=0.0,
        value=9000.0
    )

with col2:
    oldbalanceDest = st.number_input(
        "Receiver Balance Before",
        min_value=0.0,
        value=5000.0
    )

    newbalanceDest = st.number_input(
        "Receiver Balance After",
        min_value=0.0,
        value=6000.0
    )

st.write("")

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Transaction"):

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "🚨 Potential Fraud Detected\n\n"
            "This transaction has been classified as potentially fraudulent."
        )

    else:
        st.success(
            "✅ Transaction Appears Legitimate\n\n"
            "This transaction has been classified as non-fraudulent."
        )

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Fraud Detection System • Machine Learning • Scikit-learn • Streamlit"
)