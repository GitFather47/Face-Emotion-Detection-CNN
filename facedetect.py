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
    json_file = open('faceEmotionModel.json', 'r')  # Assuming your model JSON is named faceEmotionModel.json
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('faceEmotionModel.keras')  # Assuming your model weights are in faceEmotionModel.h5
    return model

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
        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame

# Main app function
def main():
    st.title("Real-time Emotion Detection")
    st.write("This app detects emotions in real-time using your camera.")
    
    # Add app description and instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'Start' button below to start the camera.
    2. The app will detect faces and show emotion predictions in real-time.
    """)

    # Start/Stop camera button
    run_camera = st.button("Start Camera")
    
    if run_camera:
        # Access the camera and display the feed
        cap = cv2.VideoCapture(0)  # 0 is the default camera
        
        frame_placeholder = st.empty()  # Placeholder for video frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from camera")
                break

            # Process the frame (detect faces and emotions)
            processed_frame = process_frame(frame)
            
            # Convert BGR to RGB for display in Streamlit
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame in the placeholder
            frame_placeholder.image(rgb_frame, channels="RGB")
            
            # Break the loop if 'Stop Camera' is pressed
            if st.button("Stop Camera"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
