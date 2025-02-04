import cv2
from keras.models import model_from_json
import numpy as np

# Load the model
json_file = open("faceEmotionModel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("faceEmotionModel.keras")

# Load the face detection model
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Webcam feed
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
colors = {
    'angry': (0, 0, 255), 'disgust': (0, 255, 255), 'fear': (255, 0, 255), 
    'happy': (0, 255, 0), 'neutral': (255, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 255, 0)
}

# Main loop for video capture and emotion detection
while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            # Predict the emotion and confidence
            pred = model.predict(img)
            pred_label_index = pred.argmax()
            prediction_label = labels[pred_label_index]
            confidence = pred[0][pred_label_index] * 100  # Get the confidence percentage

            # Set color and display label with confidence percentage
            color = colors[prediction_label]
            label_text = f"{prediction_label} ({confidence:.2f}%)"

            # Draw rectangle (box) with the same color as the predicted emotion
            cv2.rectangle(im, (p, q), (p+r, q+s), color, 2)

            # Display label on top-left of face rectangle with bold Arial-like font
            cv2.putText(im, label_text, (p, q-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 4)

        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass
