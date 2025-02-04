import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Page config
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Load model (do this only once)
@st.cache_resource
def load_model():
    json_file = open('facialemotionmodel.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('facialemotionmodel.h5')
    return model

# Load face classifier (do this only once)
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize
model = load_model()
face_classifier = load_face_classifier()
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# RTC Configuration for WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region and preprocess
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            img_features = extract_features(face_img)
            
            # Make prediction
            pred = model.predict(img_features)
            prediction_label = labels[pred.argmax()]
            
            # Display the prediction
            cv2.putText(img, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return img

def main():
    st.title("Real-time Emotion Detection")
    st.write("This app detects emotions in real-time using your camera.")
    
    # Add app description and instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'Start' button below to start the camera
    2. Allow camera access when prompted
    3. The app will detect faces and show emotion predictions
    4. Works with both webcam and mobile front camera
    """)
    
    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
