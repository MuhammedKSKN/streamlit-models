import streamlit as st
import numpy as np
import pickle
from streamlit_option_menu import option_menu

from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

tf.keras.backend.clear_session()
tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="👨‍🦰🤶")

with st.sidebar:
    selected = option_menu("Multiple Disease Prediction", 
                ['Heart Disease',
                 'Diabetes Disease',
                 'Covid-19 Detection'
                 ],
                 menu_icon='hospital-fill',
                 icons=['heart', 'activity', 'virus'],
                 default_index=0)

if selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    st.header("Enter Patient Data:")

    age = st.number_input("Age", min_value=1, max_value=80, value=30)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
    restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=0)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3, value=2)

    if st.button("Predict Heart Disease"):
        model_path = "models/heart_disease.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        input_data = np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        input_data_reshaped = input_data.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)

        if prediction[0] == 1:
            st.error("Detected Heart Disease")
        else:
            st.success("Healthy Heart")

if selected == "Diabetes Disease":
    with open('models/diabetes_model.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)

    with open('models/scaler_diabete.pkl', 'rb') as scaler_file:
        scaler_diabetes = pickle.load(scaler_file)

    st.title("Diabetes Prediction App")

    with st.form(key='diabetes_form'):
        pregnancies = st.slider('Number of Pregnancies', min_value=0, max_value=20, value=0)
        glucose = st.slider('Glucose Level', min_value=0, max_value=200, value=0)
        blood_pressure = st.slider('Blood Pressure', min_value=0, max_value=200, value=0)
        skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=0)
        insulin = st.slider('Insulin Level', min_value=0, max_value=900, value=0)
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=0.0)
        dpf = st.slider('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0)
        age = st.slider('Age', min_value=0, max_value=120, value=0)

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
            input_data = input_data.reshape(1, -1)
            input_data = scaler_diabetes.transform(input_data)

            prediction = classifier.predict(input_data)

            if prediction[0] == 0:
                st.write('Do not worry, there is no diabetes.')
            else:
                st.write('Unfortunately, there is diabetes.')

if selected == "Covid-19 Detection":
    with open('models/new_model1.pkl', 'rb') as f:
        model = pickle.load(f)

    scaler = pickle.load(open('scaler_covid/scaler.pkl', 'rb'))

    def covid_tahmini_yap(model, image, scaler):
        image = ImageOps.grayscale(image)
        image = image.resize((28, 28))
        image = np.array(image).flatten().reshape(1, -1)
        image = scaler.transform(image)
        tahmin = model.predict(image)
        return tahmin[0]

    st.title("COVID-19 Detection")

    uploaded_file = st.file_uploader("Upload a chest image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Predicting...")

        label = covid_tahmini_yap(model, image, scaler)

        if label == 0:
            st.write("Result: COVID-19 detected.")
        else:
            st.write("Result: No COVID-19 detected.")
