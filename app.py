import streamlit as st
import pandas as pd
import pickle


st.write("""
# MSDE4 : Cloud computing Course
## Diabete Prediction App

This app predicts the **diabete** 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0.0, 17.0, 2.0)
    Glucose = st.sidebar.slider('Glucose', 0.0, 199.0, 50.0)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0.0, 122.0, 20.0)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0.0, 99.0, 50.0)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 200.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 10.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction ', 0.0, 2.42, 0.5)
    Age = st.sidebar.slider('Age', 21.0, 81.0, 20.0)
    
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'Age': Age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model_app=pickle.load(open("model.pkl", "rb"))
prediction = model_app.predict(df)
prediction_proba = model_app.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(model_app.classes_))

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

