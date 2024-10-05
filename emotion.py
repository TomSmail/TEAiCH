import cv2
from mistralai import Mistral
import os 
from dotenv import load_dotenv
from datetime import datetime
import json

from utils import encode_image

# Load environment variables from .env file
load_dotenv()

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Create or overwrite the log file to store the results
with open('logs/log_file.txt', 'w') as f:
    f.write("{\n")

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the Mistral AI client
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")
print(f"API Key {api_key}")
client = Mistral(api_key=api_key)
model = "pixtral-12b-2409"

timesteps = []

class Emotion:
    def __init__(self):
        self.current_utc_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.processing = True

    def stop_processing(self):
        print("Stopping processing...")
        self.processing = False
        # Release the camera and close the log file
        camera.release()
        cv2.destroyAllWindows()

    def start_processing(self):
        while self.processing:
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
            print(f"Detected {len(faces)} faces")

            listeners = []

            current_utc_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

            for (x, y, w, h) in faces:
                # Extract the face region
                face = frame[y:y+h, x:x+w]

                # Resize the face to 224x224 pixels
                resized_face = cv2.resize(face, (224, 224))

                # Save the resized face image
                face_filename = f"./images/face_{current_utc_time}.jpg"
                os.makedirs(os.path.dirname(face_filename), exist_ok=True)
                cv2.imwrite(face_filename, resized_face)

                # Encode the image to base64
                img_base64 = encode_image(face_filename)

                # Prepare the messages for the Mistral AI client
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I want you to tell me the emotion/state displayed in this image. "
                                "I'd like you to return it as JSON in the format: "
                                "{\n"
                                "    attention_level: <attention_level>,\n"
                                "    happiness_level: <happiness_level>,\n"
                                "    sadness_level: <sadness_level>,\n"
                                "    anger_level: <anger_level>,\n"
                                "    surprise_level: <surprise_level>,\n"
                                "    confusion_level: <confusion_level>\n"
                                "}\n\n"
                                "Where each level is a float between 0 and 1. Where 0 is not present and 1 is very present."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{img_base64}" 
                        }
                    ]
                }]
                
                try:
                    # Send the request to the API using Mistral AI client
                    chat_response = client.chat.complete(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"}
                    )
                    
                    response_content = chat_response.choices[0].message.content
                    print("API Response:", response_content)

                    json_object = json.loads(response_content)

                    # Validate JSON keys
                    required_keys = [
                        "attention_level",
                        "happiness_level",
                        "sadness_level",
                        "anger_level",
                        "surprise_level",
                        "confusion_level"
                    ]
                    if not all(key in json_object for key in required_keys):
                        print("Invalid response format from API.")
                        continue

                    attention_level = json_object["attention_level"]

                    log_entry = {
                        "data": json_object
                    }
                    
                    listeners.append(log_entry)

                    # Draw a rectangle around the face and label it with the boredom level
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(
                        frame, 
                        f"Attention: {attention_level:.2f}", 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (255, 0, 0), 
                        2
                    )
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue

            # Log the predictions
            if listeners:
                timesteps.append({"timestamp": current_utc_time, "listeners": listeners})

            # Display the frame with detected faces and boredom levels
            cv2.imshow('Attention Detection', frame)

            # Check for 'Esc' key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 27 is the ASCII code for 'Esc'
                self.stop_processing()
                break

            # Optional: Adjust the sleep time as needed
            # time.sleep(2)  # Removed to allow real-time processing

    def finalize_logging(self):
        # Write the collected timesteps to the log file in JSON format
        with open('logs/log_file.txt', 'a') as log_file:
            json.dump(timesteps, log_file, indent=4)
            log_file.write("\n}")

    def main(self):
        self.start_processing()
        self.finalize_logging()

if __name__ == "__main__":
    try:
        emotion = Emotion()
        emotion.start_processing()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        emotion.stop_processing()
    finally:
        print("Processing stopped")
        print(json.dumps(timesteps, indent=4))
        emotion.finalize_logging()
