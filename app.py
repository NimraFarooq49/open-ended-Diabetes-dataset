import streamlit as st
import numpy as np
import pickle

# =====================================
# Load Trained Model
# =====================================
with open("rf_iris_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]

# =====================================
# App UI
# =====================================
st.set_page_config(
    page_title="🌸 Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Prediction")
st.markdown("### Random Forest Classifier (Iris Dataset)")
st.write("Enter flower measurements to predict the **Iris species**.")

st.divider()

# =====================================
# Input Fields
# =====================================
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1
    )
    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4
    )
    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2
    )

# =====================================
# Prediction
# =====================================
if st.button("🔮 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.success(f"🌼 **Predicted Species:** {target_names[prediction]}")

    st.markdown("### 📊 Prediction Probability")
    for i, species in enumerate(target_names):
        st.write(f"**{species}**: {prediction_proba[0][i]*100:.2f}%")

st.divider()

# =====================================
# Footer
# =====================================
st.caption("🔬 Model: Random Forest Classifier | Dataset: Iris")
st.caption("🚀 Deployed using Streamlit")
