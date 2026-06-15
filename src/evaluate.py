import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    "best_mobilenetv2.keras"
)

# =====================================
# TEST DATA
# =====================================

IMG_SIZE = 224
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(

    "dataset/chest_xray/chest_xray/test",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    color_mode="rgb",

    shuffle=False
)

# =====================================
# PREDICTIONS
# =====================================

y_pred_prob = model.predict(
    test_generator
)

y_pred = (
    y_pred_prob > 0.5
).astype(int)

y_true = test_generator.classes

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )
)