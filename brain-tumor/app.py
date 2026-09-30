# ============================================================
# Brain Tumor MRI Classification - Streamlit App
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"
]

# Get directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Model location
MODEL_PATH = BASE_DIR / "models" / "best_brain_tumor_model.keras"


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_brain_tumor_model():

    # Check whether model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found.\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )

    # Load trained Keras model
    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return loaded_model


# Load model
try:
    MODEL = load_brain_tumor_model()

except Exception as e:
    st.error("❌ Failed to load the brain tumor model.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. VERIFY MODEL
# ============================================================

if not hasattr(MODEL, "predict"):
    st.error(
        f"Loaded object is not a valid Keras model.\n\n"
        f"Loaded type: {type(MODEL)}"
    )
    st.stop()


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_image(image: Image.Image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to same size used during training
    image = image.resize(IMG_SIZE)

    # Convert PIL image to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    # (224, 224, 3)
    #       ↓
    # (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    probabilities = MODEL.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Confidence
    confidence = float(
        probabilities[predicted_index]
    )

    # Probability dictionary
    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return (
        predicted_class,
        confidence,
        probability_dict
    )


# ============================================================
# 6. HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Classification")

st.markdown(
    """
    Upload a brain MRI image and the trained deep learning
    model will classify it into one of four categories.
    """
)

st.info(
    "⚠️ This application is an AI-assisted image "
    "classification tool and is not a medical diagnosis."
)


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    st.write("**Input image size:** 224 × 224")

    st.write("**Number of classes:** 4")

    st.write("**Classes:**")

    for class_name in CLASS_NAMES:
        st.write(f"- {class_name}")

    st.write(
        f"**Model file:** `{MODEL_PATH.name}`"
    )


# ============================================================
# 8. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a JPG, JPEG, PNG, or BMP brain MRI image."
)


# ============================================================
# 9. PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.subheader("📷 Uploaded MRI Image")

    st.image(
        image,
        caption="Uploaded Brain MRI",
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # 10. PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔍 Predict Tumor Type",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing MRI image..."
            ):

                (
                    predicted_class,
                    confidence,
                    probability_dict
                ) = predict_image(image)


            # ==================================================
            # 11. RESULT
            # ==================================================

            st.subheader("🧠 Prediction Result")

            st.success(
                f"Predicted Class: "
                f"**{predicted_class.upper()}**"
            )

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}"
            )


            # ==================================================
            # 12. PROBABILITY TABLE
            # ==================================================

            st.subheader(
                "📊 Class Probability"
            )

            # Sort probabilities highest → lowest
            sorted_probabilities = dict(
                sorted(
                    probability_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

            for class_name, probability in (
                sorted_probabilities.items()
            ):

                st.write(
                    f"**{class_name.upper()}**: "
                    f"{probability:.2%}"
                )

                st.progress(
                    probability
                )


            # ==================================================
            # 13. BAR CHART
            # ==================================================

            st.subheader(
                "📈 Prediction Probability Distribution"
            )

            chart_data = {
                class_name: probability
                for class_name, probability
                in sorted_probabilities.items()
            }

            st.bar_chart(chart_data)


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Brain Tumor MRI Classification • "
    "TensorFlow / Keras + Streamlit"
)
