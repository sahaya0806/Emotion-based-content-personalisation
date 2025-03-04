import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import time

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Trained Model
model_path = "J:/kaniyakumari/emotion_model.pth"  # Update path if needed
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()

try:
    model = torch.jit.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

# Define Allowed Emotions
original_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_map = {
    "Angry": "Angry",
    "Happy": "Happy",
    "Sad": "Sad",
    "Neutral": "Neutral",
    "Disgust": "Neutral",
    "Fear": "Neutral",
    "Surprise": "Neutral"
}

# **Delay Mechanism**
last_detection_time = time.time() - 10  # Initialize 10 seconds in the past
detected_emotion = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame!")
            break  # Stop the loop if no frame is read

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        current_time = time.time()

        # **Check if 10 seconds have passed since the last detection**
        if current_time - last_detection_time < 10:
            if detected_emotion:
                cv2.putText(frame, f"Emotion: {detected_emotion} (Waiting)", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red color
            cv2.imshow('Face Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit if 'q' is pressed
            continue  # Skip detection process

        # **Reset emotion counter after 10 seconds**
        emotion_counter = {}

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]  # Extract grayscale face region

            try:
                # **Optimized Preprocessing**
                resized_face = cv2.resize(face_roi, (48, 48))  # Resize for model
                normalized_face = resized_face / 255.0  # Normalize pixel values

                # Convert to PyTorch tensor
                final_image = np.expand_dims(normalized_face, axis=(0, 1))  # Shape: (1, 1, 48, 48)
                data_tensor = torch.from_numpy(final_image).float().to(device)  # Convert NumPy to Torch

                # **Predict Emotion**
                with torch.no_grad():
                    outputs = model(data_tensor)
                    prediction = torch.argmax(F.softmax(outputs, dim=1)).item()

                # **Get the detected emotion**
                emotion = emotion_map.get(original_emotions[prediction], "Neutral")

                # Count emotion occurrences
                emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1

            except Exception as e:
                print(f"Error processing face: {e}")

        if emotion_counter:
            # **Pick the most frequent emotion**
            detected_emotion = max(emotion_counter, key=emotion_counter.get)
            print(f"Final Detected Emotion: {detected_emotion}")

            # **Display Emotion on Webcam**
            if current_time - last_detection_time >= 10:  # Only show bounding box when detecting
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw face box
                    cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Face Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

        # **Update last detection time**
        last_detection_time = time.time()

except KeyboardInterrupt:
    print("\n Program manually stopped.")

finally:
    # **Release resources and close everything safely**
    cap.release()
    cv2.destroyAllWindows()
    print(" Webcam and OpenCV windows closed gracefully.")
