import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Apex Vision Live", page_icon="👁️", layout="wide")

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; } 
    .result-card { background-color: #1c2029; padding: 20px; border-left: 5px solid #00e676; border-radius: 5px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_traffic_brain():
    paths = ["traffic_classifier.h5", "backend/traffic_classifier.h5"]
    for p in paths:
        if os.path.exists(p):
            # Compile=False prevents optimizer errors on load
            return load_model(p, compile=False)
    return None

model = load_traffic_brain()

classes = { 
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

def process_image(img):
    img = img.convert('RGB')
    img = img.resize((30, 30))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("👁️ APEX VISION // AUTONOMOUS NODE")

c1, c2 = st.columns([1, 1])

with c1:
    uploaded_file = st.file_uploader("Upload Dashcam Feed", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Sensor Input", width=300)

with c2:
    if uploaded_file and model is not None:
        if st.button("EXECUTE NEURAL SCAN", type="primary"):
            try:
                processed = process_image(image)
                pred = model.predict(processed)
                class_id = np.argmax(pred)
                confidence = float(np.max(pred)) * 100
                
                label = classes.get(class_id, "Unknown Sign")
                
                status_color = "#ff2b2b" if class_id in [14, 17] else "#00e676"
                msg = "🚨 CRITICAL ALERT" if class_id in [14, 17] else "✅ SAFE"

                st.markdown(f"""<div class="result-card" style="border-left-color: {status_color};"><h2 style="color: {status_color}; margin:0;">{msg}</h2><h1>{label.upper()}</h1></div>""", unsafe_allow_html=True)
                st.metric("CONFIDENCE", f"{confidence:.2f}%")
                
            except Exception as e:
                st.error(f"Inference Error: {e}")