import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("diamond_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("💎 Diamond Price Predictor")

st.sidebar.header("Enter Diamond Details")

# Numeric inputs
carat = st.sidebar.number_input("Carat", min_value=0.0)
depth = st.sidebar.number_input("Depth")
table = st.sidebar.number_input("Table")
x = st.sidebar.number_input("Length (x)")
y = st.sidebar.number_input("Width (y)")
z = st.sidebar.number_input("Height (z)")

# Categorical inputs
cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# Create input dataframe
input_dict = {
    'carat': carat,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z
}

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# One-hot encoding for user input
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")