####### Image training #############

import os
import numpy as np
import tensorflow as tf
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load and preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    return input_image

# Function to extract features from keypoints
def extract_features(keypoints):
    keypoints_data = keypoints.reshape([17, 3])[:, :2]
    centroid = calculate_centroid(keypoints_data)
    angles = calculate_angles(keypoints_data)
    feet_contact = check_feet_contact(keypoints_data)
    features = np.concatenate((centroid, angles, [feet_contact]))
    feature_vector = np.concatenate((keypoints_data.flatten(), features), axis=0)
    return feature_vector

# Function to calculate centroid from keypoints
def calculate_centroid(keypoints):
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return np.array([centroid_x, centroid_y])

# Function to calculate angles between keypoints
def calculate_angles(keypoints):
    left_hip, left_knee, left_ankle = keypoints[11], keypoints[12], keypoints[13]
    right_hip, right_knee, right_ankle = keypoints[8], keypoints[9], keypoints[10]
    left_leg_angle = np.arctan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0]) \
                      - np.arctan2(left_hip[1] - left_knee[1], left_hip[0] - left_knee[0])
    right_leg_angle = np.arctan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0]) \
                       - np.arctan2(right_hip[1] - right_knee[1], right_hip[0] - right_knee[0])
    left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[6], keypoints[7]
    right_shoulder, right_elbow, right_wrist = keypoints[2], keypoints[3], keypoints[4]
    left_arm_angle = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0]) \
                      - np.arctan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0])
    right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0]) \
                       - np.arctan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0])
    return np.array([left_leg_angle, right_leg_angle, left_arm_angle, right_arm_angle])

# Function to check if feet keypoints are in contact with the ground
def check_feet_contact(keypoints):
    left_foot, right_foot = keypoints[14], keypoints[11]
    if left_foot[1] > right_foot[1]:
        lower_foot = right_foot
        higher_foot = left_foot
    else:
        lower_foot = left_foot
        higher_foot = right_foot
    return lower_foot[1] <= higher_foot[1]

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='/content/lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Set paths for the 'fall' and 'nofall' directories
fall_dir = '/content/drive/MyDrive/fall'
nofall_dir = '/content/drive/MyDrive/nofall'

# Load images and labels
fall_images = [os.path.join(fall_dir, filename) for filename in os.listdir(fall_dir) if filename.endswith(('.jpg', '.png'))]
nofall_images = [os.path.join(nofall_dir, filename) for filename in os.listdir(nofall_dir) if filename.endswith(('.jpg', '.png'))]
labels = [1] * len(fall_images) + [0] * len(nofall_images)  # 1 for fall, 0 for nofall

# Combine image paths
images = fall_images + nofall_images

# Process images and extract features
features = []
for image_path in images:
    input_image = load_image(image_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    feature_vector = extract_features(keypoints_with_scores)
    features.append(feature_vector)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model
random_forest = RandomForestClassifier()
random_forest .fit(X_train, y_train)

# Evaluate the model
y_pred =random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open('random_forest.pkl', 'wb') as f:
    pickle.dump(random_forest, f)


######### Audio training #########
import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# Function to extract MFCC features from an audio file
def extract_audio_features(audio_path, sr=44100, n_mfcc=50):
    y, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Average across time
    return mfccs_mean

# Paths to your audio datasets
fall_audio_dir = '/content/drive/MyDrive/fallaudio'
nofall_audio_dir = '/content/drive/MyDrive/nofallaudio'

# Load audio files and labels
fall_audio_files = [os.path.join(fall_audio_dir, filename) for filename in os.listdir(fall_audio_dir) if filename.endswith('.wav')]
nofall_audio_files = [os.path.join(nofall_audio_dir, filename) for filename in os.listdir(nofall_audio_dir) if filename.endswith('.wav')]
labels = [1] * len(fall_audio_files) + [0] * len(nofall_audio_files)  # 1 for fall, 0 for nofall

# Combine audio file paths
audio_files = fall_audio_files + nofall_audio_files

# Extract features from the audio files
features = []
for audio_path in audio_files:
    try:
        mfccs_mean = extract_audio_features(audio_path)
        features.append(mfccs_mean)
    except Exception as e:
        print(f"Failed to process {audio_path}: {e}")

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    # 'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier(),
    # 'SVM': SVC(),
    # 'KNN': KNeighborsClassifier()
}

# Train each model and evaluate accuracy
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

# Save the best model (example: saving the Gradient Boosting model)
model_save_path = 'rf_audio_model.pkl'
with open(model_save_path, 'wb') as f:
    pickle.dump(models['Random Forest'], f)

print(f"Best model saved to {model_save_path}")
