from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import time
import cv2

app = FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Black-blight","Blister-blight","Canker","Horse-hair-blight","brown blight","healthy"]

RECOMMENDATIONS = {
    "Blister-blight": "Plant suitable clones with natural resistance. Manage shade trees by thinning or lopping. "
                      "Use chemical control methods: spraying fungicides such as copper fungicides, protective, systemic, and curative. "
                      "In nurseries, spray copper fungicides every four days (120g in 45l water). For tea recovering from pruning, "
                      "spray every 4 or 5 days (420g in 170l of water per hectare). For tea in plucking, spray once every 7-10 days "
                      "(280-420g in 170l of water per hectare).",

    "Black-blight": "Use chemical control methods: spray fungicides like copper fungicides, protective, systemic, and curative. "
                    "In heavily shaded nurseries during heavy rainfall, thin out shade and apply a 50% w/w copper fungicide spray "
                    "(120g in 45l water) with knapsack sprayers at leaf spotting onset. Repeat after two weeks if rain persists. "
                    "In new clearings, spray when stem infection appears using the same copper fungicide dilution but with approximately "
                    "430l water per hectare for thorough coverage.",

    "brown blight": "No special control measures are recommended. When outbreaks occur, copper fungicides could be used. "
                    "Control measures are not necessary in these cases. With regular spraying done for blister blight, "
                    "this disease does not pose any problem anymore.",

    "Canker": "Remove affected branches at pruning. Avoid planting tea in poor soil areas. Use vigorous plants with developed root systems "
              "for planting. Adopt proper soil conservation measures. Avoid planting susceptible cultivars in risky areas. Avoid mechanical "
              "damage to stems. Thatch soil during dry weather.",

    "Horse-hair-blight": "Remove and destroy all crop debris from around plants. Prune out infected or dead branches from the plant canopy.",

    "healthy": "This leaf is in excellent health, showing no signs of disease."
}


def resize_image(image,target_size = (512,512)):
    resized_image = cv2.resize(image,target_size)
    return resized_image

@app.get("/ping")
async def ping():
    start_time = time.time()
    try:
        return {"message": "Hello, I am alive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"Time taken to process /ping: {time.time() - start_time} seconds")

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

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    recommendation = RECOMMENDATIONS.get(predicted_class, "No recommendation found")
    return{
        'diseaseName': predicted_class,
        'confidence': float(confidence),
        'recommendation': recommendation
    }




# if __name__ == "__main__":
#     uvicorn.run(app,host='localhost',port=8000)
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
