import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Apex Vision Live",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PRO CSS (Dark Mode & Cyberpunk Aesthetics) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* Upload Box Styling */
    .stFileUploader {
        padding: 30px;
        border: 2px dashed #00e676;
        border-radius: 15px;
        background-color: #1c2029;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 30px !important;
        color: #00e676;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #fff;
    }
    
    /* Result Card */
    .result-card {
        background-color: #1c2029;
        padding: 20px;
        border-left: 5px solid #00e676;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD BRAIN (Robust Path Check) ---
@st.cache_resource
def load_traffic_brain():
    # Check possible cloud paths
    paths = ["traffic_classifier.h5", "backend/traffic_classifier.h5"]
    for p in paths:
        if os.path.exists(p):
            return load_model(p)
    return None

model = load_traffic_brain()

# --- FULL CLASS MAP (43 Classes) ---
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
    img = img.resize((30, 30))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1042/1042680.png", width=80)
    st.title("APEX CONSOLE")
    st.markdown("`v2.0 | CLOUD NODE`")
    st.markdown("---")
    st.info("💡 **SYSTEM STATUS:**\n\n🟢 NEURAL ENGINE ONLINE\n🟢 GPU ACCELERATION READY")

# --- MAIN APP ---
st.title("👁️ APEX VISION // AUTONOMOUS NODE")
st.markdown("### `REAL-TIME TRAFFIC SIGN RECOGNITION SYSTEM`")
st.markdown("---")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 1. SENSOR INPUT")
    uploaded_file = st.file_uploader("Upload Dashcam Feed / Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Raw Sensor Data", width=400)

with c2:
    st.markdown("### 2. AI DIAGNOSTICS")
    
    if uploaded_file and model is not None:
        if st.button("EXECUTE NEURAL SCAN", type="primary", use_container_width=True):
            with st.spinner("PROCESSING TENSORS..."):
                try:
                    processed = process_image(image)
                    pred = model.predict(processed)
                    class_id = np.argmax(pred)
                    confidence = float(np.max(pred)) * 100
                    
                    label = classes.get(class_id, "Unknown Object")
                    
                    # --- DYNAMIC UI ---
                    if class_id in [14, 17]: # Stop or No Entry
                        status_color = "#ff2b2b" # Red
                        status_msg = "🚨 CRITICAL ALERT"
                    else:
                        status_color = "#00e676" # Green
                        status_msg = "✅ SAFE TO PROCEED"

                    # Result Card
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color: {status_color};">
                        <h2 style="color: {status_color}; margin:0;">{status_msg}</h2>
                        <h1 style="margin-top: 10px;">{label.upper()}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    m1, m2 = st.columns(2)
                    m1.metric("CONFIDENCE", f"{confidence:.2f}%")
                    m2.metric("CLASS ID", f"#{class_id}")
                    
                    # Progress Bar
                    st.progress(int(confidence))
                        
                except Exception as e:
                    st.error(f"Inference Failed: {e}")
                    
    elif model is None:
        st.warning("⚠️ AI Model Offline. Please check GitHub storage.")
    else:
        st.info("👈 Waiting for Sensor Input...")