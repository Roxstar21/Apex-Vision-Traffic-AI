import streamlit as st
import requests
from PIL import Image
import io
import time
import random
from datetime import datetime
import streamlit.components.v1 as components

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="APEX VISION",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- JAVASCRIPT ANIMATION (Subtle Background Grid) ---
def background_animation():
    components.html("""
    <style>
        body { margin: 0; background-color: #09090b; overflow: hidden; }
        .grid {
            width: 200%; height: 200%;
            position: absolute; top: -50%; left: -50%;
            background-image: 
                linear-gradient(rgba(34, 197, 94, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34, 197, 94, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            transform: perspective(500px) rotateX(60deg);
            animation: move 20s linear infinite;
        }
        @keyframes move {
            0% { transform: perspective(500px) rotateX(60deg) translateY(0); }
            100% { transform: perspective(500px) rotateX(60deg) translateY(50px); }
        }
    </style>
    <div class="grid"></div>
    """, height=0)

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- CSS STYLING (Professional/Glassmorphism) ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp { background-color: #09090b; color: #e4e4e7; font-family: 'Inter', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Panel Styling (Glass effect) */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #22c55e;
    }
    
    /* Action Box */
    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border: none;
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        transition: transform 0.1s;
    }
    .stButton>button:active { transform: scale(0.98); }
    .stButton>button:hover { background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%); color: white; }
    
    /* Log Terminal */
    .terminal {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #a1a1aa;
        height: 150px;
        overflow-y: auto;
        border-top: 1px solid #333;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC ---
def get_action_logic(sign_class):
    if "Stop" in sign_class: return "CRITICAL STOP", "#ef4444", "🛑"
    elif "Speed" in sign_class: return "SPEED LIMIT", "#f59e0b", "⚠️"
    elif "No entry" in sign_class: return "NO ENTRY", "#ef4444", "⛔"
    elif "Turn" in sign_class: return "NAVIGATE", "#3b82f6", "↪️"
    else: return "PROCEED", "#22c55e", "✅"

# --- LAYOUT ---
background_animation()

# Top Bar (Telemetry)
c1, c2, c3, c4 = st.columns(4)
c1.metric("SYSTEM", "ONLINE", "ACTIVE")
c2.metric("GPU LOAD", f"{random.randint(20, 45)}%", "-2%")
c3.metric("LATENCY", f"{random.randint(15, 30)}ms", "STABLE")
c4.metric("CONFIDENCE THRESHOLD", "85.0%", "LOCKED")

st.markdown("---")

# Main Stage
main_col, side_col = st.columns([2, 1])

with main_col:
    st.markdown("### 📡 OPTICAL SENSOR FEED")
    
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📤 UPLOAD IMAGE", "📷 WEBCAM LINK"])
        img_bytes = None
        
        with tab1:
            u_file = st.file_uploader("Select Tensor Data", label_visibility="collapsed")
            if u_file:
                img_bytes = u_file.getvalue()
                # FIXED: use_container_width instead of use_column_width
                st.image(Image.open(u_file), use_container_width=True) 

        with tab2:
            c_file = st.camera_input("Live Feed", label_visibility="collapsed")
            if c_file:
                img_bytes = c_file.getvalue()
                
        st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    st.markdown("### 🧠 INTELLIGENCE STACK")
    
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        
        if img_bytes:
            if st.button("RUN INFERENCE PROTOCOL", type="primary"):
                with st.spinner("Analyzing Tensor Data..."):
                    time.sleep(0.5) # Cinematic delay
                    try:
                        response = requests.post(API_URL, files={"file": img_bytes})
                        
                        if response.status_code == 200:
                            data = response.json()
                            sign = data["sign_class"]
                            conf = float(data["confidence"].strip('%'))
                            action, color, icon = get_action_logic(sign)
                            
                            # Log
                            st.session_state['history'].insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {sign} | {conf}% | {action}")
                            
                            # Result Card
                            st.markdown(f"""
                            <div style="text-align: center; margin-bottom: 20px;">
                                <div style="font-size: 3rem; margin-bottom: 10px;">{icon}</div>
                                <h2 style="margin: 0; color: white;">{sign}</h2>
                                <p style="color: #a1a1aa; font-size: 0.9rem;">CONFIDENCE: <span style="color: {color}">{conf}%</span></p>
                            </div>
                            <div class="status-indicator" style="background-color: {color}20; color: {color}; border: 1px solid {color}40;">
                                <span>ACTION REQ</span>
                                <span>{action}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(int(conf))
                            
                        else:
                            st.error("SERVER HANDSHAKE FAILED")
                    except Exception as e:
                        st.error(f"CONNECTION ERROR: {e}")
        else:
            st.info("Awaiting Data Stream...")
            st.markdown("""
            <div style="margin-top: 20px; font-size: 0.8rem; color: #555;">
                SECURE CONNECTION ESTABLISHED<br>
                MODEL: TRAFFIC_RESNET_V4<br>
                BUILD: 2026.02.09
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Footer Logs
st.markdown("### 📟 SYSTEM EVENTS")
log_history = "<br>".join(st.session_state['history'])
st.markdown(f"""
<div class="glass-panel terminal">
    > APEX_VISION initialized...<br>
    > Monitoring Port 8000...<br>
    {log_history}
</div>
""", unsafe_allow_html=True)