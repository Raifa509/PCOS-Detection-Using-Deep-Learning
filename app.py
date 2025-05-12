import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load Models
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("pcos_xgb_model.pkl")

# Initialize session state
if 'show_main' not in st.session_state:
    st.session_state.show_main = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_advice' not in st.session_state:
    st.session_state.show_advice = False

# Custom CSS for Styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: white !important;
        color: black !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: 2px solid black !important;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: black !important;
        color: white !important;
    }

    .lifestyle-btn > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        transition: 0.3s;
    }

    .lifestyle-btn > button:hover {
        background-color: #218838 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome Page
def welcome_page():
    st.markdown("<h1>🩺 OvaCare Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h2>PCOS Detection System</h2>", unsafe_allow_html=True)

    welcome_image = Image.open(r"C:\Users\npfra\OneDrive\Desktop\Main project\code\bg2.jpg")
    st.image(welcome_image, width=500, use_container_width=False)

    if st.button("Get Started ➔"):
        st.session_state.show_main = True
        st.rerun()

# Main Application UI
def main_ui():
    st.markdown("<h1>PCOS Detection System Using Deep Learning</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is None:
        st.session_state.prediction = None
        st.session_state.show_advice = False

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Predict", key='predict_button', help="Click to predict PCOS status"):
                img_array = np.array(image.convert('RGB'))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_resized = cv2.resize(img_array, (224, 224))
                img_preprocessed = preprocess_input(img_resized)
                img_expanded = np.expand_dims(img_preprocessed, axis=0)

                features = vgg16_model.predict(img_expanded).flatten().reshape(1, -1)
                features_scaled = scaler.transform(features)

                prediction = xgb_model.predict(features_scaled)
                
                st.session_state.prediction = "PCOS Detected" if prediction[0] == 1 else "Normal"
                st.session_state.show_advice = False
                st.rerun()

    with col2:
        if st.session_state.prediction is not None:
            st.subheader("Prediction Result")
            st.write(f"**{st.session_state.prediction}**")

            if st.session_state.prediction == "PCOS Detected":
                if st.button("Lifestyle Tips", key="lifestyle_btn", help="Click for lifestyle recommendations"):
                    st.session_state.show_advice = True
                    st.rerun()

            if st.session_state.show_advice:
                st.write("- **Eat a balanced diet**")
                st.write("- **Reduce sugar intake**")
                st.write("- **Engage in regular exercise**")
                st.write("- **Stay hydrated** and aim for at least 7-9 hours of sleep per night.")

# Page Routing
if not st.session_state.show_main:
    welcome_page()
else:
    main_ui()
