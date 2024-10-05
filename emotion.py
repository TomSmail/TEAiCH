import cv2
import numpy as np
import tensorflow as tf
import time

# Load the emotion recognition model
model = tf.keras.models.load_model('emotion_model.h5')

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a log file to store the results
log_file = open('logs/log_file.txt', 'a')

# Initialize the camera
camera = cv2.VideoCapture(0)

categories = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    print('Capturing frame...')
    # Capture a frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame")
        break

    print(f"Frame shape: {frame.shape}")

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face to 224x224 pixels
        resized_face = cv2.resize(face, (224, 224))

        # Normalize the face (e.g., scale pixel values to [0, 1])
        normalized_face = resized_face / 255.0

        # Expand dimensions to match the model input shape (1, 224, 224, 3)
        input_face = np.expand_dims(normalized_face, axis=0)

        # Make predictions using the model
        predictions = model.predict(input_face)

        # Get the predicted emotion
        predicted_emotion = categories[np.argmax(predictions)]

        # Log the predictions
        log_file.write(f"Face at ({x}, {y}, {w}, {h}): {predicted_emotion}\n")
        log_file.flush()

        print(f"Face at ({x}, {y}, {w}, {h}): {predicted_emotion}")
        # Optionally, draw a rectangle around the face and label it with the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detected faces and predicted emotions
    cv2.imshow('Emotion Recognition', frame)

    # Wait for 10 seconds
    time.sleep(10)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the log file
camera.release()
log_file.close()
cv2.destroyAllWindows()