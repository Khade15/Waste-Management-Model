import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('waste management_model.pkl')

# Function to make predictions
def predict_management_strategy(waste_type, material_composition, recycling_potential, toxicity_level):
    features = np.array([[waste_type, material_composition, recycling_potential, toxicity_level]])
    prediction = model.predict(features)
    return '♻️ Recyclable' if prediction[0] == 1 else '🗑️ Non-Recyclable'

# Streamlit app
def main():
    st.title("♻️ Waste Management Strategy Predictor 🗑️")

    st.header("Input Waste Characteristics:")

    waste_type = st.selectbox("🗂️ Waste Type:", [1, 2, 3, 4, 5])
    material_composition = st.slider("🔬 Material Composition:", 0.0, 1.0, 0.5)
    recycling_potential = st.slider("♻️ Recycling Potential:", 0.0, 1.0, 0.5)
    toxicity_level = st.slider("☠️ Toxicity Level:", 0.0, 100.0, 50.0)

    if st.button("🔍 Predict Management Strategy"):
        result = predict_management_strategy(waste_type, material_composition, recycling_potential, toxicity_level)
        st.success(f"The predicted management strategy is: {result}")

if __name__ == "__main__":
    main()
