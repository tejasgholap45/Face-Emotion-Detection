import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
import urllib.request

# =====================================================
# AUTO DOWNLOAD MODEL
# =====================================================
MODEL_URL = "https://drive.google.com/uc?id=18oDZKqtFDJ9j2pItGEYZG415OiA21Ck4"
MODEL_PATH = "model_file.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =====================================================
# AUTO DOWNLOAD HAARCASCADE
# =====================================================
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("❌ Face detection model failed to load.")
    st.stop()

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">😊 Face Emotion Detection</h1>
    <p style="text-align:center; font-size:18px;">
    AI-powered emotion recognition using <b>TensorFlow & CNN</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# =====================================================
# MAIN UI
# =====================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    st.markdown("### 🧠 Detected Emotion")
    st.info("Upload an image to see results")

# =====================================================
# PROCESS IMAGE
# =====================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1.image(image, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    except Exception:
        st.error("❌ Face detection failed.")
        st.stop()

    if len(faces) == 0:
        st.warning("⚠️ No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = roi.reshape(1, 48, 48, 1)

            preds = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img_array,
                f"{emotion} ({confidence:.1f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        col2.success(f"### 😊 Emotion Detected: **{emotion}**")
        col2.progress(int(confidence))

        st.markdown("### 📊 Detection Result")
        st.image(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

# =====================================================
# FOOTER (ONLY ONCE)
# =====================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:14px;">
    <b>Developed by Tejas Gholap</b><br>
    🔗 <a href="https://www.linkedin.com/in/tejas-gholap" target="_blank">LinkedIn</a> |
    💻 <a href="https://github.com/tejasgholap45" target="_blank">GitHub</a><br><br>

    <b>Guided By:</b> Pratik Ramteke Sir<br>
    <b>Powered By:</b> Cloudblitz | Powered by Greamio<br><br>

    <i>TensorFlow-based Face Emotion Detection | Streamlit Web App</i>
    </div>
    """,
    unsafe_allow_html=True
)
