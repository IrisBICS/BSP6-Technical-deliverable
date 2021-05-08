import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATAPATH = "../FER2013/icml_face_data.csv"
LANDMARKSPATH = "../FER2013/fer2013_landmarks.csv"
SAVELOCIMG = "../FER2013/images/"
SAVELOCLMKIMG = "../FER2013/landmark_images/"
SAVELOCOTHER = "../FER2013/other/"

EMOTIONDATA = pd.read_csv(DATAPATH, sep=r'\s*,\s*', engine='python')

def generateBalancingPlot(emotion_data, save_loc, labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')):

    counts = emotion_data["emotion"].astype(int)
    plt.hist(counts, bins=range(8), align='left')
    plt.xlabel("Emotion index")
    plt.ylabel("Count")
    plt.title("Emotion categories distribution")
    plt.xticks(range(len(labels)), labels)
    plt.savefig(save_loc + "balance.png", bbox_inches='tight')
    plt.show()

def getAndFormatImage(data, index, gray=True, padding=0):

    pixels = data.iloc[index]['pixels'].split(" ")
    array = np.array([int(p) for p in pixels], dtype=np.uint8)
    img = array.reshape((48, 48))
    img = np.pad(img, padding)

    if gray:
        img = np.stack([img, img, img], axis=-1)

    return img

def displayImages(emotion_data, save_loc, indexes=range(10), labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')):

    for i in indexes:
        emotion = emotion_data.iloc[i]['emotion']
        emotion_cat = labels[emotion]
        img = getAndFormatImage(emotion_data, i)

        plt.imshow(img)
        plt.title("Image " + str(i) + " - Labeled " + emotion_cat)
        plt.axis('off')
        plt.savefig(save_loc + str(i) + ".png", bbox_inches='tight')
        plt.show()

def splitLmkData(data):

    points = data.split(" ")
    xy_split = [point.split(".") for point in points]

    float_xy = []
    for [x, y] in xy_split:
        float_xy.append([float(x), float(y)])

    return float_xy

def displayLandmarksOnImages(emotion_data, lmk_path, save_loc, indexes=range(10), labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')):

    landmarks_data = pd.read_csv(lmk_path, sep=r'\s*,\s*', engine='python')
    padding = 3

    for i in indexes:

        img = getAndFormatImage(emotion_data, i, padding=padding)
        plt.imshow(img)
        emotion = emotion_data.iloc[i]['emotion']
        emotion_cat = labels[emotion]

        row = landmarks_data.iloc[i]
        data = []

        for part in ["jaw", "right_brow", "left_brow", "nose", "right_eye", "left_eye", "mouth", "lips"]:
            data += splitLmkData(row[part])

        for [x,y] in data:
            plt.scatter(x+padding, y+padding, color='r')

        plt.title("Image " + str(i) + " - Labeled " + emotion_cat)
        plt.axis('off')
        plt.savefig(save_loc + str(i) + ".png", bbox_inches='tight')
        plt.show()


#generateBalancingPlot(EMOTIONDATA, SAVELOCOTHER)
#displayImages(EMOTIONDATA, SAVELOCIMG)
#displayLandmarksOnImages(EMOTIONDATA, LANDMARKSPATH, SAVELOCLMKIMG)