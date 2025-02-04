import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Page config
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Load model (do this only once)
@st.cache_resource
def load_model():
    try:
        json_file = open('faceEmotionModel.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights('faceEmotionModel.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load face classifier (do this only once)
@st.cache_resource
def load_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize
model = load_model()
face_classifier = load_face_classifier()
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to process and detect emotion in each frame
def process_frame(frame):
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region and preprocess
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        img_features = extract_features(face_img)
        
        # Make prediction
        pred = model.predict(img_features)
        prediction_label = labels[pred.argmax()]
        
        # Display the prediction
        cv2.putText(frame, prediction_label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame

# Main app function
def main():
    st.title("Real-time Emotion Detection")
    
    # Check model loading
    if model is None:
        st.error("Failed to load emotion detection model. Please check model files.")
        return
    
    st.write("This app detects emotions in real-time using your camera.")
    
    # Camera selection
    camera_options = ['Default Camera (0)', 'External Camera (1)']
    camera_choice = st.selectbox("Select Camera", camera_options)
    camera_index = 0 if camera_choice == 'Default Camera (0)' else 1
    
    # Start/Stop camera
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error(f"Unable to open camera {camera_index}. Please check camera connection.")
            return
        
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Camera")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Process the frame
            processed_frame = process_frame(frame)
            
            if processed_frame is not None:
                # Convert BGR to RGB for display in Streamlit
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB")
            
            # Check stop button again
            stop_button = st.button("Stop Camera")
        
        # Release the camera
        cap.release()
        st.success("Camera stopped")

if __name__ == "__main__":
    main()
