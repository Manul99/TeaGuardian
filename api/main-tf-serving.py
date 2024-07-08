from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import requests
app = FastAPI()

#MODEL = tf.keras.models.load_model("../saved_models/1")
endpoint = "http://localhost:8502/v1/models/tea_model:predict"
CLASS_NAMES = ["Black-blight","Blister-blight","Canker","Horse-hair-blight","brown blight","healthy"]

# Resizing the image
def resize_image(image,target_size = (512,512)):
    resized_image = cv2.resize(image,target_size)
    return resized_image

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    print("Image Shape:", image.shape)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    orginal_image = read_file_as_image(await file.read())
    print("File Name:", file.filename)
    print("Content Type:", file.content_type)
    print("File Size (bytes):", len(await file.read()))

    resized_image = resize_image(orginal_image)
    print("Resized Image shape: ",resized_image.shape)


    img_batch = np.expand_dims(resized_image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)