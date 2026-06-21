# Spoken Digit Recognition using Deep Learning | Kaggle Competition

## Overview

This project was developed for a Kaggle competition conducted as part of the EE708 – Fundamentals of Data Science and Machine Intelligence course at IIT Kanpur. The objective was to build a deep learning model capable of recognizing spoken digits (0–9) from short audio clips. The challenge focused on developing a model that could generalize well to unseen speakers and noisy recordings rather than memorizing speaker-specific characteristics.

## Problem Statement

Given a short `.wav` audio clip containing a spoken digit, the task is to classify it into one of the ten digit classes (0–9).

The main challenges were:
- Different speakers in the training and test sets
- Background noise and recording variations
- Short-duration audio clips
- Building a model that generalizes well instead of memorizing the training data

## Approach

### Audio Preprocessing

- Loaded audio files using Librosa
- Resampled all audio to 16 kHz
- Fixed every audio clip to exactly one second using padding or center trimming
- Converted each waveform into a Mel Spectrogram
- Created a 3-channel input consisting of:
  - Log-Mel Spectrogram
  - Delta
  - Delta-Delta

### Data Augmentation

To improve robustness, multiple augmentation techniques were used:

- Random noise injection
- Random time shifting
- Frequency masking
- Time masking (SpecAugment)
- Mixup
- Label smoothing

These techniques helped the model perform well on noisy recordings and unseen speakers.

### Model

The model is based on ResNet-18 with a few modifications for audio spectrograms.

Changes made:
- Trained from scratch without pretrained weights
- Modified the first convolution layer for spectrogram inputs
- Removed the initial max-pooling layer
- Replaced the final classification head with custom fully connected layers and dropout

### Training

The model was trained using:

- Cross-Entropy Loss with Label Smoothing
- AdamW Optimizer
- Cosine Annealing Learning Rate Scheduler
- Stratified K-Fold Cross Validation
- Mixed Precision Training

The best model from each fold was saved and used during inference.

## Results

- Out-of-Fold Validation Accuracy: **99.54%**
- 10-class spoken digit classification
- Strong performance on unseen speakers and noisy audio

## Technologies Used

- Python
- PyTorch
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Repository Structure

```text
Spoken-Digit-Recognition/
│
├── Presentation.pdf                  # Project presentation
├── README.md                         # Project documentation
├── Report_DR.pdf                     # Detailed project report
└── Spoken_Digit_Recognition.ipynb    # Complete training and inference pipeline
```


## Dataset

The dataset was provided through the Kaggle competition and consisted of:

- train_audio/ containing labeled .wav audio files
- test_audio/ containing unlabeled .wav audio files
- train.csv with audio IDs and corresponding digit labels
- test.csv containing audio IDs for prediction

The test labels were hidden during the competition, and model performance was evaluated based on predictions submitted to Kaggle.

Since the dataset was provided exclusively for the competition, it cannot be redistributed and is therefore not included in this repository.
## Skills Demonstrated

- Audio Signal Processing
- Deep Learning
- Convolutional Neural Networks (CNNs)
- Feature Engineering
- Data Augmentation
- Model Evaluation
- PyTorch

## Contributors

- Bipin Kumar Jaiswal
- Yash Kumar
- Dhruv Bajaj
- Anshika Singh
- Peeyush Sahu

## Acknowledgements

This project was completed as part of the EE708 - Fundamentals of Data Science and Machine Intelligence course at IIT Kanpur.
