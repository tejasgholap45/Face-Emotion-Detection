import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# ---------------- auto download ----------------
MODEL_URL = "https://drive.google.com/uc?id=18oDZKqtFDJ9j2pItGEYZG415OiA21Ck4"
MODEL_PATH = "model_file.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("😊 Face Emotion Detection using TensorFlow")
st.write(
    "A professional **Streamlit web application** for detecting "
    "**human emotions from facial expressions** using a CNN model."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_emotion_model():
    return load_model("model_file.h5")

model = load_emotion_model()

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

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

            cv2.rectangle(
                img_array, (x, y), (x+w, y+h), (0, 255, 0), 2
            )
            cv2.putText(
                img_array, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        st.subheader("Detection Result")
        st.image(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

# ---------------- FOOTER (ONLY ONCE) ----------------
st.markdown("---")
st.markdown(
    """
    **Developed by Tejas Gholap**  
    🔗 LinkedIn: https://www.linkedin.com/in/tejas-gholap  
    💻 GitHub: https://github.com/tejasgholap45  

    **Guided By:** Pratik Ramteke Sir  
    **Powered By:** Cloudblitz | Powered by Greamio  

    *TensorFlow-based Face Emotion Detection | Streamlit Web App*
    """
)
