import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import WebRtcMode,webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Page config
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Load model (do this only once)
@st.cache_resource
def load_model():
    json_file = open('faceEmotionModel.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('faceEmotionModel.keras')
    return model

# Load face classifier (do this only once)
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize
model = load_model()
face_classifier = load_face_classifier()
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# RTC Configuration for WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract and preprocess face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Make prediction
            predictions = model.predict(roi)[0]
            
            # Display emotion percentages on right side
            bar_width = 150
            for i, (emotion, pred) in enumerate(zip(emotions, predictions)):
                # Calculate percentage
                percentage = int(pred * 100)
                
                # Draw text and bar
                text_x = img.shape[1] - bar_width - 10
                text_y = 30 + i * 30
                
                # Draw emotion label
                cv2.putText(img, f'{emotion}:', (text_x - 80, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw percentage bar
                bar_x = text_x
                bar_y = text_y - 10
                filled_width = int((percentage * bar_width) / 100)
                
                # Draw background bar
                cv2.rectangle(img, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + 10), 
                             (120, 120, 120), -1)
                
                # Draw filled bar
                cv2.rectangle(img, (bar_x, bar_y), 
                             (bar_x + filled_width, bar_y + 10), 
                             (65, 105, 225), -1)
                
                # Draw percentage text
                cv2.putText(img, f'{percentage}%', 
                           (bar_x + bar_width + 10, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img

def main():
    st.title("Real-time Emotion Detection")
    st.write("This app detects emotions in real-time using your camera.")
    
    # Add app description and instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'Start' button below to start the camera
    2. Allow camera access when prompted
    3. The app will detect faces and show emotion percentages
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
