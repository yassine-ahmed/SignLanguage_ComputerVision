# Sign Language Computer Vision Project

## Overview
This project aims to develop a computer vision system that can recognize sign language gestures using a webcam. The system captures images of hand gestures, processes them to extract hand landmarks, and then classifies these gestures using a machine learning model.

## Project Structure
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