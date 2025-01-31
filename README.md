# realtime_fall

This repository contains an implementation of a fall detection system that utilizes the Movenet algorithm to estimate the poses of people in a live video stream. The extracted key points are then used to derive features, and fed into a gradient booster machine learning algorithm for fall identification. Additionally, the system captures the video and uploads it to an Amazon S3 bucket for storage and further analysis.

## Introduction
This project aims to detect falls in a real-time video stream using a combination of computer vision techniques and machine learning. The Movenet algorithm is employed to estimate the poses of individuals, and these pose keypoints are utilized to derive features that enable the detection of falls. We achieve a robust fall detection system by integrating a gradient booster machine learning algorithm.

## System Architecture
The fall detection system follows the following high-level architecture:

1. Video Input: The system takes input from a live video stream or recorded video files.
2. Movenet Pose Estimation: The Movenet algorithm is applied to estimate the poses of individuals in each frame of the video.
3. Feature Extraction: Key points from the estimated poses are extracted and transformed into features suitable for fall detection.
4. Gradient Booster Algorithm: The extracted features are fed into a gradient booster machine learning algorithm to classify falls.
5. Fall Detection: The gradient booster algorithm identifies falls based on the extracted features.
6. Video Capture: If a fall is detected, the system captures the relevant video segment.
7. S3 Bucket Upload: The captured video segment is uploaded to an Amazon S3 bucket for storage and further analysis.

### Software and Tools requirement

1. [Github Account](https://github.com)
2. [VSCode](https://code.visualstudio.com/)
3. [HerokuAccount](https://heroku.com)

Create a new environment

'''
conda create -p venv python==3.7 -y

'''

Install the required dependencies.
'''
pip install -r requirements.txt

'''

## Configuration
VIDEO_SOURCE: Set this to specify the source of the video stream (e.g., camera device index or video file path).
S3_BUCKET_NAME: Set this to the name of your Amazon S3 bucket where the captured video segments will be uploaded.
AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY: Set these with your AWS access key and secret access key, respectively, to authorize the upload to the S3 bucket.



