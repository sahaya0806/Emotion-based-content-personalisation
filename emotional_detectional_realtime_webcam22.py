import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import webbrowser

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Trained Model
model_path = "J:/kaniyakumari/emotion_model.pth"  # Update path if needed
if not os.path.exists(model_path):
    print("Model not found. Exiting...")
    exit()

try:
    model = torch.jit.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model. Exiting...")
    exit()

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

# Define Allowed Emotions
original_emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Paths to the HTML files
html_files = {
    "Happy": "http://127.0.0.1:5500/happy.html",  # Happy page
    "Sad": "http://127.0.0.1:5500/sad.html",  # Sad page
    "Neutral": "http://127.0.0.1:5500/neutral.html",  # Neutral page
    "Angry": "http://127.0.0.1:5500/angryaddedpart.html",  # Angry page
}

# **Wait for 10 seconds to analyze the face expression**
analysis_duration = 10  # Duration in seconds
start_time = time.time()
detected_emotion = None

# Create OpenCV window
cv2.namedWindow('Face Emotion Recognition', cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop the loop if no frame is read

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        # **Calculate elapsed time and remaining time**
        elapsed_time = time.time() - start_time
        time_remaining = max(0, analysis_duration - int(elapsed_time))

        # **Display the detected emotion and countdown timer**
        if detected_emotion:
            cv2.putText(frame, f"Emotion: {detected_emotion} (Waiting)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red color
        cv2.putText(frame, f"Next Detection In: {time_remaining}s", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)  # Yellow color

        # **Analyze emotions only during the 10-second window**
        if elapsed_time < analysis_duration:
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
                    emotion = original_emotions[prediction]  # Use the original emotion labels

                    # **Map unwanted emotions (Surprise, Fear, Disgust) to Neutral**
                    if emotion in ["Surprise", "Fear", "Disgust"]:
                        emotion = "Neutral"  # Map to Neutral

                    # **Store the detected emotion**
                    detected_emotion = emotion

                except Exception as e:
                    pass  # Ignore errors during face processing

        # **Display the webcam feed**
        cv2.imshow('Face Emotion Recognition', frame)
        cv2.waitKey(1)  # Ensure the window updates

        # **Break the loop after 10 seconds**
        if elapsed_time >= analysis_duration:
            break

        # **Exit if 'q' is pressed**
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass  # Handle manual interruption gracefully

finally:
    # **Release resources and close everything safely**
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows closed gracefully.")

# **After 10 seconds, determine the final emotion and redirect**
if detected_emotion in html_files:
    print(f"Final Detected Emotion: {detected_emotion}")
    print(f"{detected_emotion} detected! Opening the page...")
    webbrowser.open(html_files[detected_emotion])  # Open the corresponding HTML file
else:
    print("No emotion detected or emotion not mapped to a page.")