from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

import shutil

from predict import predict_image

app = FastAPI()

@app.get("/")
def home():

    return {

        "message":
        "Pneumonia Detection API Running"
    }

@app.post("/predict")
async def predict(

    file: UploadFile = File(...)
):

    file_path = file.filename

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_image(
        file_path
    )

    return result