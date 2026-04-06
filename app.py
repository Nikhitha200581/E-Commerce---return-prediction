import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🛒 E-Commerce Return Prediction")

# User inputs

product_category = st.number_input("Product Category (encoded)", min_value=0)
delivery_time = st.slider("Delivery Time (days)", 1, 10)
rating = st.slider("Product Rating", 1, 5)
previous_returns = st.slider("Previous Returns", 0, 10)
payment_method = st.selectbox("Payment Method", ["COD", "Card"])


# Convert inputs
payment_method = 0 if payment_method == "COD" else 1

# Create dataframe
input_data = pd.DataFrame([{
    'Product_Category': product_category,
    'Delivery_Time': delivery_time,
    'Rating': rating,
    'Previous_Returns': previous_returns,
    'Payment_Method': payment_method,
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Product will be RETURNED")
    else:
        st.success("✅ Product will NOT be returned")