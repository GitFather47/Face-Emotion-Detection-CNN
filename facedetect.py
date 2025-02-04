import cv2
from keras.models import model_from_json
import numpy as np
import streamlit as st
from PIL import Image

# Load the model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Streamlit app
st.title("Real-Time Facial Emotion Detection")

# Start the webcam
webcam = cv2.VideoCapture(0)

# Placeholder for the video frame
frame_placeholder = st.empty()

# Loop to capture video frames
while True:
    ret, frame = webcam.read()
    if not ret:
        st.error("Failed to capture video from webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # Process each face detected
    for (p, q, r, s) in faces:
        # Extract the face region
        face_image = gray[q:q + s, p:p + r]
        face_image = cv2.resize(face_image, (48, 48))

        # Extract features and predict emotion
        img = extract_features(face_image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the frame in the Streamlit app
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
