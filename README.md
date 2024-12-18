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



### 2. Creating the Dataset `create_dataset.py`
This script processes the collected images to extract hand landmarks using the MediaPipe library and prepares the dataset for training.

#### Key Features:
- Reads images from the collected `/data` directory.
- Uses MediaPipe to detect hand landmarks and normalize their coordinates (key points of the hand).
- Saves the processed data and corresponding labels into a pickle file.

#### Key Code:
- Detect hand landmarks:
```bash
results = hands.process(img_rgb)
```
- Normalize and store the coordinates:
```bash
data_aux.append(x - min(x_))
data_aux.append(y - min(y_))
```

#### Outcome:
- A `pickle` file (`data.pickle`) is created, containing features (landmark vectors) and their corresponding class labels.


#### Usage:
Run the script in a terminal:
```bash
python create_dataset.py
```



### 3. Training the Model `train_classifier.py`
This script trains a Random Forest classifier using the dataset created in the previous step.


#### Key Features:
- Loads the dataset from the (`data.pickle`) file.
- Splits the data into training and testing sets.
- A `RandomForestClassifier` is trained on the landmark data.
- Saves the trained model to a `model.p` file.

#### Key Code:
- Train the model:
```bash
model.fit(x_train, y_train)
```
- Save the model:
```bash
pickle.dump({'model': model}, f)
```

#### Outcome:
- A trained Random Forest model is stored in `model.p`.

#### Usage:
Run the script in a terminal:
```bash
python train_classifier.py
```




### 4. Real-Time Gesture Recognition `inference_classifier.py`
This script performs real-time gesture recognition using the trained model.


#### Key Features:
- The trained model (`model.p`) is loaded.
- The webcam captures frames in real-time.
- MediaPipe detects hand landmarks in each frame.
- The trained model predicts the gesture based on the landmarks.
- The predicted gesture is displayed on the webcam feed.

#### Key Code:
- Load the model:
```bash
model_dict = pickle.load(open('./model.p', 'rb'))
```
- Predict gesture:
```bash
prediction = model.predict([np.asarray(data_aux)])
```
- Display prediction:
```bash
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
```

#### Outcome:
- The system recognizes hand gestures and displays the results on the video feed​.



#### Usage:
Run the script in a terminal:
```bash
python inference_classifier.py
```
## Data Directory
The `/data` directory contains subdirectories for each class (0, 1, 2) with images collected during the data collection phase. Each image is saved in JPEG format.


# Features

## Core Functionalities
- Real-time gesture recognition using webcam input.
- Customizable classes for gesture training.
- Lightweight processing with MediaPipe for real-time performance.

## Potential Improvements
- **Scalability**: Add database support for managing gesture datasets.
- **Feature Enhancements**:
  - Save and analyze unrecognized gestures.
  - Develop a REST API for remote gesture recognition.
- **Performance Optimization**:
  - Utilize threading to accelerate dataset creation and model training.

## Troubleshooting
- **Camera Not Detected**: Verify the correct camera index in OpenCV's `VideoCapture` method.
- **Low Recognition Accuracy**: Use higher-quality images and adjust frame resizing for better results.
- **Dependency Issues**: Ensure all required Python libraries are installed.

## Acknowledgments
This project leverages the following libraries:
- **OpenCV**: For video capture and image processing.
- **MediaPipe**: For hand landmark detection.
- **scikit-learn**: For training and evaluating the machine learning model.

For further details:
- [OpenCV Documentation](https://opencv.org/)
- [MediaPipe GitHub](https://github.com/google/mediapipe)
- [scikit-learn Documentation](https://scikit-learn.org/)
