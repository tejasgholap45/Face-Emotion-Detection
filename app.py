import streamlit as st
from ultralytics import YOLO
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Live Face Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("😊 Live Face Emotion Detection using YOLOv11")
st.write(
    "Real-time **face emotion detection** using a custom-trained "
    "**YOLOv11 model** with Streamlit."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained emotion model

model = load_model()

# ---------------- WEBCAM CONTROL ----------------
run = st.checkbox("▶️ Start Live Webcam")
frame_window = st.image([])
cap = cv2.VideoCapture(0)

# ---------------- LIVE DETECTION ----------------
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Unable to access webcam")
        break

    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    frame_window.image(annotated_frame)

cap.release()

# ---------------- FOOTER (ONLY ONCE) ----------------
st.markdown("---")
st.markdown(
    """
    **Developed by Tejas Gholap**  
    🔗 LinkedIn: https://www.linkedin.com/in/tejas-gholap  
    💻 GitHub: https://github.com/tejasgholap45  

    **Guided By:** Pratik Ramteke Sir  
    **Powered By:** Cloudblitz | Powered by Greamio  

    *YOLOv11-based Live Face Emotion Detection | Streamlit Web App*
    """
)
