import dlib
import pandas as pd
import numpy as np


DATAPATH = "../FER2013/icml_face_data.csv"
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("../dlibModels/shape_predictor_68_face_landmarks.dat")
LANDMARKSPATH = "../FER2013/fer2013_landmarks.csv"


def getImageAtIndex(data, i=0):

    pixels = data.iloc[i]['pixels'].split(" ")
    array = np.array([int(p) for p in pixels], dtype=np.uint8)
    img = array.reshape((48, 48))

    return img


def getFacialLandmarks(img, faceIsCropped=True):

    if faceIsCropped:  # Meaning image is already just a cropped face
        face = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
    else:
        faces = DETECTOR(img)
        face = faces[0]

    landmarks = PREDICTOR(image=img, box=face)

    landmarks_dict = {}

    landmarks_dict["jaw"] = formatLandmarks(landmarks, 0, 16)
    landmarks_dict["right_brow"] = formatLandmarks(landmarks, 17, 21)
    landmarks_dict["left_brow"] = formatLandmarks(landmarks, 22, 26)
    landmarks_dict["nose"] = formatLandmarks(landmarks, 27, 35)
    landmarks_dict["right_eye"] = formatLandmarks(landmarks, 36, 41)
    landmarks_dict["left_eye"] = formatLandmarks(landmarks, 42, 47)
    landmarks_dict["mouth"] = formatLandmarks(landmarks, 48, 60)
    landmarks_dict["lips"] = formatLandmarks(landmarks, 61, 67)

    return landmarks_dict

def formatLandmarks(landmarks, low, high, xy_sep=".", pnt_sep=" "):

    formatted = ""
    for i in range(low, high+1):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        formatted += str(x) + xy_sep + str(y)
        if i < high:
            formatted += pnt_sep

    return formatted


def generateLandmarksCsv(name):

    emotion_data = pd.read_csv(DATAPATH, sep=r'\s*,\s*', engine='python')

    df = pd.DataFrame()

    for _, row in emotion_data.iterrows():
        pixels = row['pixels'].split(" ")
        array = np.array([int(p) for p in pixels], dtype=np.uint8)
        img = array.reshape((48, 48))

        landmarks = getFacialLandmarks(img)
        landmarks["emotion"] = row["emotion"]
        df = df.append(landmarks, ignore_index=True)

    with open(name, mode='w') as csv_file:
        df.to_csv(csv_file, index=False, line_terminator='\n')

#generateLandmarksCsv(LANDMARKSPATH)