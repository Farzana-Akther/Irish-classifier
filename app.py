import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Title of the app
st.title("Iris Flower Prediction App")

# Input fields for the model
sepal_length = st.number_input("Sepal Length:", min_value=0.0)
sepal_width = st.number_input("Sepal Width:", min_value=0.0)
petal_length = st.number_input("Petal Length:", min_value=0.0)
petal_width = st.number_input("Petal Width:", min_value=0.0)

# Prediction button
if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    result = model.predict(input_features)

    # Display the result
    st.success(f"The predicted class is: {result[0]}")
