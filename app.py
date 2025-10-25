import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('sample_house_prices.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Train model
X = df[['squarefootage', 'bedrooms', 'bathrooms', 'quality', 'garage', 'basement']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🏠 House Price Prediction using Linear Regression")

sqft = st.number_input("Square Footage")
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
quality = st.slider("Quality (1-10)", 1, 10)
garage = st.selectbox("Garage", [0, 1])
basement = st.selectbox("Basement", [0, 1])

if st.button("Predict Price"):
    features = [[sqft, bedrooms, bathrooms, quality, garage, basement]]
    price = model.predict(features)
    st.success(f"Estimated House Price: ₹{price[0]:,.2f}")