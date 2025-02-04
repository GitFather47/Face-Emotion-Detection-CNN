import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

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

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Main app function
def main():
    st.title("Real-time Emotion Detection")
    st.write("This app detects emotions in real-time using your camera.")
    
    # App instructions
    st.markdown("""
    ### Instructions:
    1. Click 'Start Detection' to open your webcam.
    2. Allow camera access when prompted.
    3. The app will detect your face and predict your emotions in real-time.
    """)
    
    run = st.checkbox('Start Detection')
    FRAME_WINDOW = st.image([])

    # OpenCV video capture
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to access webcam.")
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract the face region and preprocess it for the model
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            img_features = extract_features(face_img)
            
            # Predict the emotion
            pred = model.predict(img_features)
            prediction_label = labels[pred.argmax()]
            
            # Display the emotion label
            cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Convert color from BGR to RGB for displaying in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Release the camera when the detection stops
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
