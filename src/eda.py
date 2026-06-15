import os
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/chest_xray/chest_xray"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")
VAL_PATH = os.path.join(DATASET_PATH, "val")

NORMAL_PATH = os.path.join(TRAIN_PATH, "NORMAL")
PNEUMONIA_PATH = os.path.join(TRAIN_PATH, "PNEUMONIA")


def show_images(folder, title):

    image_files = [
        img for img in os.listdir(folder)
        if img.lower().endswith((".jpeg", ".jpg", ".png"))
    ][:5]

    plt.figure(figsize=(15, 5))

    for i, image_name in enumerate(image_files):

        image_path = os.path.join(folder, image_name)

        image = Image.open(image_path)

        plt.subplot(1, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


show_images(NORMAL_PATH, "Normal X-Rays")
show_images(PNEUMONIA_PATH, "Pneumonia X-Rays")