import tensorflow as tf
import numpy as np
import cv2
import os
import datetime
import pickle
import librosa
import pyaudio
from twilio.rest import Client

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directory to save images
save_folder = 'C:/sightica/saved_images'
os.makedirs(save_folder, exist_ok=True)

# Define parameters for SMS sending
NUM_CONSECUTIVE_FRAMES = 10  # Increased for more robust detection
CONFIDENCE_THRESHOLD = 0.70  # Raised for higher certainty before triggering
AUDIO_CONFIDENCE_THRESHOLD = 0.75  # Set threshold for audio model
COOLDOWN_FRAMES = 50
consecutive_fall_count = 0
cooldown_counter = 0

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Send SMS alert
def send_sms_alert():
    acc_sid = 'AC371a2cc277d0a872d1a92da9935ae4e7'
    auth_token = '881b671677e9cb0d0d1705c30cb91846'
    client = Client(acc_sid, auth_token)
    message = client.messages.create(
        body="Person fallen",
        from_='+15707018763',
        to='+917356259401'
    )
    print("SMS sent.")


# Helper functions for feature extraction
def calculate_centroid(keypoints):
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return np.array([centroid_x, centroid_y])

def calculate_angles(keypoints):
    def compute_angle(a, b, c):
        return np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    left_leg_angle = compute_angle(keypoints[11], keypoints[12], keypoints[13])
    right_leg_angle = compute_angle(keypoints[8], keypoints[9], keypoints[10])
    left_arm_angle = compute_angle(keypoints[5], keypoints[6], keypoints[7])
    right_arm_angle = compute_angle(keypoints[2], keypoints[3], keypoints[4])

    return np.array([left_leg_angle, right_leg_angle, left_arm_angle, right_arm_angle])

def check_feet_contact(keypoints):
    left_foot = keypoints[14]
    right_foot = keypoints[11]
    return left_foot[1] <= right_foot[1]

def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Load the pre-trained video-based classifier
with open('C:/Users/Amala/extracted_path/realtime_fall-main/random_forest_good.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the pre-trained audio model
with open('rf_audio_model.pkl', 'rb') as f:
    audio_model = pickle.load(f)

# Setup PyAudio for real-time audio processing
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Video capture setup
cap = cv2.VideoCapture(0)

# Real-time processing
audio_data = np.array([], dtype=np.float32)  # Initialize empty array for audio data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    keypoints = keypoints_with_scores.reshape([17, 3])
    keypoints_data = keypoints[:, :2]

    centroid = calculate_centroid(keypoints_data)
    angles = calculate_angles(keypoints_data)
    feet_contact = check_feet_contact(keypoints_data)

    features = np.concatenate((centroid, angles, [feet_contact]))
    feature_vector = np.concatenate((keypoints_data.flatten(), features), axis=0)
    x_test1 = feature_vector.reshape(1, -1)

    video_pred = clf.predict(x_test1)
    video_confidence = clf.predict_proba(x_test1)[:, 1][0]  # Extract video confidence

    audio_pred = 0

    # Function to calculate RMS energy of audio data
    def is_silent(audio_data, threshold=0.012):
        rms = np.sqrt(np.mean(audio_data**2))  # Root Mean Square (RMS) energy
        return rms < threshold

    # Real-time audio processing
    data = stream.read(1024)
    data_np = np.frombuffer(data, dtype=np.float32)
    audio_data = np.concatenate((audio_data, data_np), axis=0)  # Append new audio data
    
    # Check for fall even if there is silence
    silent_fall = False
    if len(audio_data) >= 44100:  # Process once we have enough audio data (1 second)
        try:
            if is_silent(audio_data):
                print("Silent fall detected.")
                silent_fall = True  # Flag for silent fall detection
            else:
                mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=50)
                audio_features = np.mean(mfccs.T, axis=0)
                x_test2 = audio_features.reshape(1, -1)

                audio_pred = audio_model.predict(x_test2)
                audio_pred_proba = audio_model.predict_proba(x_test2)

                print(f"Audio model prediction: {audio_pred}")
                print(f"Audio model prediction probability: {audio_pred_proba}")

                audio_data = np.array([], dtype=np.float32)  # Reset buffer

        except Exception as e:
            print(f"Error during audio processing: {e}")

    # Implement decision logic for both audio and video fall detection or silent fall
    if video_pred == 1:
        print(f'Fall detected (Video Confidence: {video_confidence:.2f})')
         # Display "Fall detected" text on the frame
        cv2.putText(
            frame,  # Frame to draw on
            'Fall detected',  # Text to display
            (50, 50),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale (size)
            (0, 0, 255),  # Color (Red)
            2,  # Thickness
            cv2.LINE_AA  # Line type for better rendering
        )

        # If video confidence is above threshold OR silent fall detected, trigger SMS
        if video_confidence >= CONFIDENCE_THRESHOLD or silent_fall:
            if cooldown_counter == 0 and silent_fall:
                print("Sending SMS alert for fall detection...")
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                image_filename = f'fall_{timestamp}.jpg'
                image_path = os.path.join(save_folder, image_filename)
                cv2.imwrite(image_path, frame)
                print(f'Saved image: {image_path}')
                send_sms_alert()
                cooldown_counter = COOLDOWN_FRAMES
        else:
            consecutive_fall_count = 0  # Reset if no fall detected
    else:
        consecutive_fall_count = 0  # Reset if confidence is not enough

    draw_keypoints(frame, keypoints, 0.4)
    draw_connections(frame, keypoints, EDGES, 0.4)
    cv2.imshow('Fall Detection', frame)

    if cooldown_counter > 0:
        cooldown_counter -= 1  # Decrease cooldown counter

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import datetime
# import pickle
# import librosa
# import pyaudio
# from twilio.rest import Client
# import time
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Directory to save images
# save_folder = 'C:/sightica/saved_images'
# os.makedirs(save_folder, exist_ok=True)

# # Define parameters for SMS sending
# NUM_CONSECUTIVE_FRAMES = 10  # Increased for more robust detection
# CONFIDENCE_THRESHOLD = 85  # Raised for higher certainty before triggering
# AUDIO_CONFIDENCE_THRESHOLD = 0.80  # Set threshold for audio model
# COOLDOWN_FRAMES = 50
# consecutive_fall_count = 0
# cooldown_counter = 0

# # Load the TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
# interpreter.allocate_tensors()

# # Send SMS alert
# def send_sms_alert():
#     acc_sid = 'AC371a2cc277d0a872d1a92da9935ae4e7'
#     auth_token = '881b671677e9cb0d0d1705c30cb91846'
#     client = Client(acc_sid, auth_token)
#     message = client.messages.create(
#         body="Person fallen",
#         from_='+15707018763',
#         to='+917356259401'
#     )
#     print("SMS sent.")

# # Helper functions for feature extraction
# def calculate_centroid(keypoints):
#     x_coords = keypoints[:, 0]
#     y_coords = keypoints[:, 1]
#     centroid_x = np.mean(x_coords)
#     centroid_y = np.mean(y_coords)
#     return np.array([centroid_x, centroid_y])

# def calculate_angles(keypoints):
#     def compute_angle(a, b, c):
#         return np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
#     left_leg_angle = compute_angle(keypoints[11], keypoints[12], keypoints[13])
#     right_leg_angle = compute_angle(keypoints[8], keypoints[9], keypoints[10])
#     left_arm_angle = compute_angle(keypoints[5], keypoints[6], keypoints[7])
#     right_arm_angle = compute_angle(keypoints[2], keypoints[3], keypoints[4])

#     return np.array([left_leg_angle, right_leg_angle, left_arm_angle, right_arm_angle])

# def check_feet_contact(keypoints):
#     left_foot = keypoints[14]
#     right_foot = keypoints[11]
#     return left_foot[1] <= right_foot[1]

# def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
#     y, x, _ = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# EDGES = {
#     (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
#     (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
#     (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
#     (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
#     (12, 14): 'c', (14, 16): 'c'
# }

# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, _ = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]

#         if (c1 > confidence_threshold) and (c2 > confidence_threshold):
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# # Load the pre-trained video-based classifier
# with open('C:/Users/Amala/extracted_path/realtime_fall-main/random_forest_good.pkl', 'rb') as f:
#     clf = pickle.load(f)

# # Load the pre-trained audio model
# with open('rf_audio_model.pkl', 'rb') as f:
#     audio_model = pickle.load(f)

# # Setup PyAudio for real-time audio processing
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# # Camera initialization function
# def initialize_camera(retries=5, delay=2):
#     cap = None
#     for attempt in range(retries):
#         cap = cv2.VideoCapture(0)
#         if cap.isOpened():
#             print("Camera connected successfully.")
#             return cap
#         else:
#             print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
#             cap.release()
#             time.sleep(delay)
#     print("Failed to connect to the camera after several attempts.")
#     return None

# # Initialize camera with retries
# cap = initialize_camera()
# if cap is None:
#     print("Exiting program due to camera connection failure.")
#     exit()

# # Real-time processing loop
# audio_data = np.array([], dtype=np.float32)  # Initialize empty array for audio data

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         # Camera reconnection logic
#         cap.release()
#         cap = initialize_camera()
#         continue #ing if no frame is captured

#     img = frame.copy()
#     img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
#     input_image = tf.cast(img, dtype=tf.float32)

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
#     interpreter.invoke()
#     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

#     keypoints = keypoints_with_scores.reshape([17, 3])

#     # Check if a person is detected by verifying confidence scores
#     confidence_scores = keypoints[:, 2]  # Extract confidence scores for all keypoints
#     max_confidence = np.max(confidence_scores)
#     print(f"Max confidence score in frame: {max_confidence:.2f}")

#     # If the highest confidence score is below the threshold, skip this frame
#     if max_confidence < 0.5:  # Adjust this threshold as needed
#         print("No person detected in the frame.")
#         # Optionally, you can reset counters or flags here
#         consecutive_fall_count = 0
#         silent_fall = False
#         continue

#     keypoints_data = keypoints[:, :2]

#     centroid = calculate_centroid(keypoints_data)
#     angles = calculate_angles(keypoints_data)
#     feet_contact = check_feet_contact(keypoints_data)

#     features = np.concatenate((centroid, angles, [feet_contact]))
#     feature_vector = np.concatenate((keypoints_data.flatten(), features), axis=0)
#     x_test1 = feature_vector.reshape(1, -1)

#     video_pred = clf.predict(x_test1)
#     video_confidence = clf.predict_proba(x_test1)[:, 1][0]

#     audio_pred = 0
#     silent_fall = False  # Reset silent fall flag at the start of each frame

#     # Function to calculate RMS energy of audio data
#     def is_silent(audio_data, threshold=0.012):
#         rms = np.sqrt(np.mean(audio_data**2))  # Root Mean Square (RMS) energy
#         return rms < threshold

#     # Real-time audio processing
#     try:
#         data = stream.read(1024)
#         data_np = np.frombuffer(data, dtype=np.float32)
#         audio_data = np.concatenate((audio_data, data_np), axis=0)

#         if len(audio_data) >= 44100:  # Process audio every second
#             if is_silent(audio_data):
#                 print("Silent fall detected.")
#                 silent_fall = True  # Flag for silent fall detection
#             else:
#                 mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=50)
#                 audio_features = np.mean(mfccs.T, axis=0)
#                 x_test2 = audio_features.reshape(1, -1)

#                 audio_pred = audio_model.predict(x_test2)
#                 audio_pred_proba = audio_model.predict_proba(x_test2)

#                 print(f"Audio model prediction: {audio_pred}")
#                 print(f"Audio model prediction probability: {audio_pred_proba}")

#             audio_data = np.array([], dtype=np.float32)  # Reset buffer

#     except Exception as e:
#         print(f"Error during audio processing: {e}")

#    # If a fall is detected
#     if video_pred == 1 and (video_confidence > CONFIDENCE_THRESHOLD or silent_fall):
#         if cooldown_counter == 0:
#             print("Fall Detected - Sending SMS alert...")

#             # Display the "Fall Detected" text on the video feed
#             cv2.putText(frame, 'Fall Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
#                         2, (0, 0, 255), 3, cv2.LINE_AA)

#             # Save the frame where fall is detected
#             timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#             image_filename = f'fall_{timestamp}.jpg'
#             image_path = os.path.join(save_folder, image_filename)
#             cv2.imwrite(image_path, frame)

#             send_sms_alert()
#             cooldown_counter = COOLDOWN_FRAMES
#             consecutive_fall_count = 0
#         else:
#             cooldown_counter -= 1
#     else:
#         consecutive_fall_count = 0

#     # Draw keypoints and connections on the frame
#     draw_connections(frame, keypoints_with_scores, EDGES, 0.5)  # Adjusted confidence threshold
#     draw_keypoints(frame, keypoints_with_scores, 0.5)  # Adjusted confidence threshold

#     # Display the frame with keypoints, connections, and the fall detected message
#     cv2.imshow('Fall Detection System', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# stream.stop_stream()
# stream.close()
# p.terminate()
# cv2.destroyAllWindows()