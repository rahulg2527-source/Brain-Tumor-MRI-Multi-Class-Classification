import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"
]


def predict_image(model, image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": float(
            probabilities[predicted_index]
        ),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
    }