import cv2
import numpy as np
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import streamlit as st

# Load the model
json_file = open("faceEmotionModel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("faceEmotionModel.keras")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Custom VideoTransformer class for processing video frames
class EmotionDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each face detected
        for (p, q, r, s) in faces:
            # Extract the face region
            face_image = gray[q:q + s, p:p + r]
            face_image = cv2.resize(face_image, (48, 48))

            # Extract features and predict emotion
            img_features = extract_features(face_image)
            pred = model.predict(img_features)
            prediction_label = labels[pred.argmax()]

            # Draw a rectangle around the face and display the predicted emotion
            cv2.rectangle(img, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(img, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        return img

# Streamlit app
def main():
    st.title("Real-Time Facial Emotion Detection with Streamlit and WebRTC")

    # Add a description
    st.write("This app uses a pre-trained CNN model to detect facial emotions in real-time.")

    # Start the WebRTC streamer
    webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=EmotionDetectionTransformer,
        async_transform=True,
    )

if __name__ == "__main__":
    main()
