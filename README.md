# Sign Language Computer Vision Project

## Overview
This project aims to develop a computer vision system that can recognize sign language gestures using a webcam. The system captures images of hand gestures, processes them to extract hand landmarks, and then classifies these gestures using a machine learning model.

## Project Structure

```
/SignLanguage_ComputerVision_Project
│
├── collect_imgs.py          # Script to collect images for training
├── create_dataset.py        # Script to create a dataset from collected images
├── inference_classifier.py   # Script for real-time gesture recognition
├── train_classifier.py      # Script to train the machine learning model
├── model.p                  # Trained model file
├── data.pickle              # Pickled dataset containing features and labels
└── /data                    # Directory containing collected images
    ├── /0                  # Images for class 0
    ├── /1                  # Images for class 1
    └── /2                  # Images for class 2
```

## Prerequisites
Before running the project, ensure the following:

1. Python 3.7 or above is installed.
2. Required libraries are installed:
   - OpenCV
   - MediaPipe
   - NumPy
   - scikit-learn
   - Matplotlib (optional, for visualization)
   
You can install these with:
```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib
```
3. A working webcam or external camera for capturing gestures.

## Scripts

### 1. Collecting Images `collect_imgs.py`
This script is responsible for collecting images of hand gestures. It captures images from the webcam and saves them in a specified directory.

#### Key Features:
- The script uses a webcam to capture hand gesture images.
- Images are stored in the `/data` directory, organized by class labels (e.g., `0`, `1`, `2`).
- Displays a message on the webcam feed to indicate when to capture images.

#### Key Code:
- Webcam feed is captured using OpenCV:
```bash
cap = cv2.VideoCapture(0)
```
- Images are saved in respective folders:
```bash
cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
```

#### Outcome:
- A dataset of images for different gesture classes is created in `/data`

#### Usage:
Run the script in a terminal:
```bash
python collect_imgs.py
```
Press `Q` to start capturing images for each class.

### 2. `create_dataset.py`
This script processes the collected images to extract hand landmarks using the MediaPipe library and prepares the dataset for training.

#### Key Features:
- Reads images from the collected data directory.
- Uses MediaPipe to detect hand landmarks and normalize their coordinates.
- DSaves the processed data and corresponding labels into a pickle file.

#### Usage:
Run the script in a terminal:
```bash
python create_dataset.py
```
### 3. `train_classifier.py`
This script trains a Random Forest classifier using the dataset created in the previous step.


#### Key Features:
- Loads the dataset from the pickle file.
- Splits the data into training and testing sets.
- Trains a Random Forest model and evaluates its accuracy.
- Saves the trained model to a file.

#### Usage:
Run the script in a terminal:
```bash
python train_classifier.py
```
### 4. `inference_classifier.py`
This script performs real-time gesture recognition using the trained model.


#### Key Features:
- Captures video from the webcam.
- Processes each frame to detect hand landmarks.
- Uses the trained model to predict the gesture being made.
- Displays the predicted gesture on the webcam feed.

#### Usage:
Run the script in a terminal:
```bash
python inference_classifier.py
```
## Data Directory
The `/data` directory contains subdirectories for each class (0, 1, 2) with images collected during the data collection phase. Each image is saved in JPEG format.