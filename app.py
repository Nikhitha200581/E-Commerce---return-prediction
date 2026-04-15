import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("model.pkl")


# Load dataset
df = pd.read_csv("balanced_ecommerce_returns_dataset-1.csv")

# Create Return_Status column
df['Return_Status'] = df['Returned'].map({0: 'Not Returned', 1: 'Returned'})

st.title("🛒 E-Commerce Return Prediction")


# User inputs

product_category = st.number_input("Product Category (encoded)", min_value=0)
# 2. Category Chart
# -----------------------------
fig2, ax2 = plt.subplots()
sns.barplot(x='Product_Category', y='Returned', data=df, ax=ax2)
ax2.set_title("Returns by Product Category")
st.pyplot(fig2)

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


if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    not_return_prob = prob[0] * 100
    return_prob = prob[1] * 100

    # Show result
    if prediction == 1:
        st.error(f"⚠️ Product will be RETURNED ({return_prob:.2f}%)")
    else:
        st.success(f"✅ Product will NOT be returned ({not_return_prob:.2f}%)")

    # -----------------------------
    # PIE CHART (DYNAMIC)
    # -----------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.pie(
        [return_prob, not_return_prob],
        labels=["Returned", "Not Returned"],
        autopct='%1.1f%%'
    )
    ax.set_title("Prediction Probability")

    st.pyplot(fig)
