import cv2
from mistralai import Mistral
import time
import os 
from dotenv import load_dotenv
from datetime import datetime
import json

from utils import encode_image

# Load environment variables from .env file
load_dotenv()

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a log file to store the results
log_file = open('logs/log_file.txt', 'a')

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the Mistral AI client
api_key = os.environ["MISTRAL_API_KEY"]
print(f"API Key {api_key}")
client = Mistral(api_key=api_key)
model = "pixtral-12b-2409"
    

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
    print (f"Detected {len(faces)} faces")

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face to 224x224 pixels
        resized_face = cv2.resize(face, (224, 224))

        # Encode the face image as a JPEG
        _, img_encoded = cv2.imencode('.jpg', resized_face)

        print(f"Type of img_encoded: {type(img_encoded)}")
        
        current_utc_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        face_filename = f"./images/face_{current_utc_time}.jpg"
        cv2.imwrite(face_filename, resized_face)

        # Encode the image to base64
        img_base64 = encode_image(face_filename)

        # Send the request to the API using Mistral AI client
        messages = [{
        "role": "user",
        "content": 
            [
                {
                    "type": "text",
                    "text": 
                    """
                        I want you to tell me the emotion/state displayed in this image, I'd like you to return it as JSON in the format: 
                            {
                                boredom_level: <boredom_level>,
                                happiness_level: <happiness_level>,
                                sadness_level: <sadness_level>,
                                anger_level: <anger_level>,
                                surprise_level: <surprise_level>,
                                confusion_level: <confusion_level>
                            }

                        Where each level is a float between 0 and 1. Where 0 is not present and 1 is very present.
                    """
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_base64}" 
                }
            ]
        }]
        
        chat_response = client.chat.complete(
                model = model,
                messages = messages,
                response_format = {
                    "type": "json_object",
                }
            )
        print(f"Chat Response: {chat_response}")
        print(chat_response.choices[0].message.content)

        json_object = json.loads(chat_response.choices[0].message.content)

        boredom_level = json_object["boredom_level"]

        # Log the predictions
        log_file.write(f"{current_utc_time}: Data: {json_object}\n")
        log_file.flush()

        # Optionally, draw a rectangle around the face and label it with the boredom level
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Boredom: {boredom_level}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detected faces and boredom levels
    cv2.imshow('Boredom Detection', frame)

    # Wait for 10 seconds
    time.sleep(2)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release the camera and close the log file
camera.release()
log_file.close()
cv2.destroyAllWindows()