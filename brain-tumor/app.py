import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"
]

MODEL_PATH = Path("models/best_brain_tumor_model.keras")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    MODEL_PATH = Path("models/best_brain_tumor_model.keras")


model = load_model()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    return predicted_class, confidence, probabilities


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload a brain MRI image to classify it into one of "
    "four categories."
)

st.info(
    "This application is an AI-assisted image classification "
    "tool and is not a medical diagnosis."
)


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    if st.button("🔍 Predict Tumor Type"):

        with st.spinner("Analyzing MRI image..."):

            predicted_class, confidence, probabilities = (
                predict_image(image)
            )


        st.subheader("Prediction")

        st.success(
            f"Predicted Class: **{predicted_class.upper()}**"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


        # --------------------------------------------------
        # Probability Distribution
        # --------------------------------------------------

        st.subheader("Class Probabilities")

        probability_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        st.bar_chart(probability_dict)