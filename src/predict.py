import os

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 224

# ==========================================================
# LOAD MODEL
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_mobilenetv2.keras"
)

model = load_model(
    MODEL_PATH
)

print("Model Loaded Successfully!")

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(img_path):

    img = image.load_img(

        img_path,

        target_size=(
            IMG_SIZE,
            IMG_SIZE
        )
    )

    img_array = image.img_to_array(
        img
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(

        img_array,

        axis=0
    )

    return img_array

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_image(img_path):

    img_array = preprocess_image(
        img_path
    )

    prediction = model.predict(

        img_array,

        verbose=0
    )[0][0]

    if prediction >= 0.5:

        label = "PNEUMONIA"

        confidence = prediction * 100

    else:

        label = "NORMAL"

        confidence = (
            1 - prediction
        ) * 100

    return {

        "prediction": label,

        "confidence": round(
            float(confidence),
            2
        )
    }

# ==========================================================
# LOCAL TESTING
# ==========================================================

if __name__ == "__main__":

    image_path = input(
        "Enter Image Path: "
    )

    result = predict_image(
        image_path
    )

    print("\nPrediction Result:")
    print(result)