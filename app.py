import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌼 Iris Flower Classifier")
st.write("Input flower features to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Input dataframe
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)',
                                   'petal length (cm)', 'petal width (cm)'])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: **{species[prediction]}**")
    
    # Probability bar chart
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(species, probability, color=['#4CAF50', '#2196F3', '#FF5722'])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    st.write("### Input Values")
    st.dataframe(input_data)

