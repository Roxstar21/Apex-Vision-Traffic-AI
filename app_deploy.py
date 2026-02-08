import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Apex Vision Live", page_icon="👁️", layout="wide")

# --- CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stFileUploader { padding: 20px; border: 2px dashed #444; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; }
</style>
""", unsafe_allow_html=True)

# --- LOAD BRAIN ---
@st.cache_resource
def load_traffic_brain():
    # Check all possible paths
    paths = ["traffic_classifier.h5", "backend/traffic_classifier.h5"]
    for p in paths:
        if os.path.exists(p):
            return load_model(p)
    return None

model = load_traffic_brain()

# --- CLASS MAP (Full GTSRB Dataset) ---
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
    # Resize to 30x30 pixels (Must match training input)
    img = img.resize((30, 30))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

# --- MAIN APP ---
st.title("👁️ APEX VISION // AUTONOMOUS NODE")
st.markdown("`NEURAL NETWORK: CONVOLUTIONAL (CNN) | ACCURACY: ~96%`")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 1. UPLOAD SENSOR DATA")
    uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Feed", width=300)

with c2:
    st.markdown("### 2. AI DIAGNOSTICS")
    if uploaded_file and model is not None:
        if st.button("RUN INFERENCE SCAN", type="primary"):
            with st.spinner("Analyzing Tensor Data..."):
                try:
                    processed = process_image(image)
                    pred = model.predict(processed)
                    class_id = np.argmax(pred)
                    confidence = float(np.max(pred)) * 100
                    
                    label = classes.get(class_id, "Unknown Sign")
                    
                    # Result Card
                    st.success(f"DETECTED OBJECT: {label.upper()}")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("CONFIDENCE SCORE", f"{confidence:.2f}%")
                    m2.metric("CLASS ID", f"#{class_id}")
                    
                    # Warning logic for critical signs
                    if class_id in [14, 17]: # Stop or No Entry
                        st.error("🚨 CRITICAL ALERT: IMMEDIATE STOP REQUIRED")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    
    elif model is None:
        st.warning("⚠️ AI Model Offline. Please check GitHub storage.")