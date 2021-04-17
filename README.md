# BSP6-Technical-deliverable

This repository contains the code of the technical deliverable of my 6th Bachelor Semester Project.

## Datasets used

* [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## External models used

* [shape_predictor_68_face_landmarks](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) to extract the facial landmarks of the FER2013 images

## Trained models

Please find the neural networks trained with the code of this repository [here](https://drive.google.com/drive/folders/1tRyeVec0-Ih8gHUIzojiWgUEJAR2AM6l?usp=sharing). For each model:

* The JSON file contains the architecture.
* The h5 file contains the weights.
* The csv file contains the training history.
* The plots folder contains the plots of:
  1. Training and validation accuracy
  2. Training and validation loss
  3. Learning rate
  4. Confusion matrices for training, validation and testing data
