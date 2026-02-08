import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Apex Vision Live", page_icon="👁️", layout="wide")

st.markdown("""<style>.stApp { background-color: #0e1117; } .result-card { background-color: #1c2029; padding: 20px; border-left: 5px solid #00e676; border-radius: 5px; margin-top: 20px; }</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_traffic_brain():
    paths = ["traffic_classifier.h5", "backend/traffic_classifier.h5"]
    for p in paths:
        if os.path.exists(p): return load_model(p)
    return None

model = load_traffic_brain()

classes = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
    33:'Turn right ahead', 34:'Turn left ahead' 
}

def process_image(img):
    # FIX 1: FORCE RGB (Removes Alpha Channel/Transparency)
    img = img.convert('RGB')
    
    # FIX 2: FORCE EXACT RESIZE
    img = img.resize((30, 30))
    img = np.array(img)
    
    # FIX 3: NORMALIZE (Just in case model expects it)
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
                
                label = classes.get(class_id, "Traffic Sign")
                
                if class_id in [14, 17]: status_color = "#ff2b2b"; msg = "🚨 CRITICAL ALERT"
                else: status_color = "#00e676"; msg = "✅ SAFE"

                st.markdown(f"""<div class="result-card" style="border-left-color: {status_color};"><h2 style="color: {status_color}; margin:0;">{msg}</h2><h1>{label.upper()}</h1></div>""", unsafe_allow_html=True)
                st.metric("CONFIDENCE", f"{confidence:.2f}%")
                
            except Exception as e:
                st.error(f"Error: {e}")