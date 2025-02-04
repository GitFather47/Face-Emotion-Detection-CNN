import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json

# Load the pre-trained model
@st.cache_resource
def load_model():
    json_file = open("faceEmotionModel.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("faceEmotionModel.keras")
    return model

# Load Haar cascade for face detection
@st.cache_resource
def load_face_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

def main():
    st.title("Facial Emotion Recognition")
    
    # Emotion labels
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
              4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    # Load model and face cascade
    model = load_model()
    face_cascade = load_face_cascade()
    
    # Webcam input
    run = st.checkbox('Open Webcam')
    FRAME_WINDOW = st.image([])
    
    if run:
        webcam = cv2.VideoCapture(0)
        
        while run:
            i, im = webcam.read()
            if not i:
                break
            
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(im, 1.3, 5)
            
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                
                cv2.putText(im, prediction_label, (p-10, q-10), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            # Convert BGR to RGB for Streamlit display
            FRAME_WINDOW.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        
        webcam.release()
    else:
        st.write('Stopped')

if __name__ == "__main__":
    main()
