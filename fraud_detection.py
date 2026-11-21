import streamlit as st 
import pandas as pd
import joblib

model = joblib.load("models/fraud_detection_pipeline.pkl")

st.title("Aplikasi Prediksi Deteksi Penipuan")

st.markdown("Silakan masukkan detail transaksi dan gunakan tombol prediksi")


st.divider()


transaction_type = st.selectbox("Jenis Transaksi", ["PAYMENT", "TRANSFER", "CASH_OUT"])
amount = st.number_input("Jumlah Transaksi", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Pengirim)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Pengirim)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Penerima)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Penerima)", min_value=0.0, value=0.0)

if st.button("Prediksi"):
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"Transaksi ini diprediksi sebagai PENIPUAN dengan probabilitas {prediction_proba[0][1]:.2f}")
    else:
        st.success(f"Transaksi ini diprediksi sebagai AMAN dengan probabilitas {prediction_proba[0][0]:.2f}")

