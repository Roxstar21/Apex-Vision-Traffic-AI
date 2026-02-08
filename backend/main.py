from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import io
import numpy as np
import tensorflow as tf

MODEL_PATH = "traffic_classifier.h5"
CLASSES = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
    41:'End of no passing', 42:'End no passing veh > 3.5 tons' 
}

app = FastAPI(title="High-Res Traffic API")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ High-Res Model Loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    if model is None: return {"error": "Model loading..."}
    
    image = Image.open(io.BytesIO(await file.read()))
    
    # CRITICAL: Must match training size (60x60)
    image = image.resize((60, 60)) 
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions)
    confidence = float(np.max(predictions))
    
    return {
        "sign_class": CLASSES.get(class_id, "Unknown"),
        "confidence": f"{confidence * 100:.2f}%",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)