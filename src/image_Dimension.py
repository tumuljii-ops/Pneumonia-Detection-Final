from PIL import Image
import os

DATASET_PATH = "dataset/chest_xray/chest_xray/train/NORMAL"

images = [
    img for img in os.listdir(DATASET_PATH)
    if img.endswith((".jpeg", ".jpg", ".png"))
]

for image_name in images[:10]:

    image_path = os.path.join(DATASET_PATH, image_name)

    image = Image.open(image_path)

    print(
        image_name,
        image.size,
        image.mode
    )