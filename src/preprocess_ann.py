import os
import numpy as np
from PIL import Image

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE = 150

DATASET_PATH = "dataset/chest_xray/chest_xray/train"

NORMAL_PATH = os.path.join(DATASET_PATH, "NORMAL")
PNEUMONIA_PATH = os.path.join(DATASET_PATH, "PNEUMONIA")

# ==========================================================
# FEATURES AND LABELS
# ==========================================================

X = []
y = []

# ==========================================================
# FUNCTION TO LOAD IMAGES
# ==========================================================

def load_images(folder_path, label):

    for image_name in os.listdir(folder_path):

        if not image_name.lower().endswith(
            (".jpeg", ".jpg", ".png")
        ):
            continue

        image_path = os.path.join(
            folder_path,
            image_name
        )

        try:

            # Force grayscale
            image = Image.open(image_path).convert("L")

            # Resize
            image = image.resize(
                (IMG_SIZE, IMG_SIZE)
            )

            # Convert to numpy
            image = np.array(
                image,
                dtype=np.float32
            )

            # Normalize
            image = image / 255.0

            X.append(image)
            y.append(label)

        except Exception as e:

            print(
                f"Error loading {image_name}: {e}"
            )

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading NORMAL images...")
load_images(NORMAL_PATH, 0)
print("NORMAL images loaded successfully!")

print("Loading PNEUMONIA images...")
load_images(PNEUMONIA_PATH, 1)
print("PNEUMONIA images loaded successfully!")

# ==========================================================
# CONVERT TO NUMPY ARRAYS
# ==========================================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nBefore Flattening")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==========================================================
# FLATTEN FOR ANN
# ==========================================================

X = X.reshape(X.shape[0], -1)

print("\nAfter Flattening")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==========================================================
# VERIFY
# ==========================================================

print("\nDataset Summary")
print("Total Samples:", len(X))
print("Total Labels :", len(y))

print("\nSample Feature Vector Shape:")
print(X[0].shape)

print("\nFirst Label:")
print(y[0])

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

os.makedirs("processed_data", exist_ok=True)

np.save("processed_data/X_train.npy", X_train)
np.save("processed_data/X_test.npy", X_test)

np.save("processed_data/y_train.npy", y_train)
np.save("processed_data/y_test.npy", y_test)