import pandas as pd
import numpy as np

DATAPATH = "../FER2013/icml_face_data.csv"
WEIGHTSPATH = "../FER2013/fer2013_weights.csv"

def getWeights(name):

    emotion_data = pd.read_csv(DATAPATH, sep=r'\s*,\s*', engine='python')
    emotion_data["emotion"] = pd.to_numeric(emotion_data["emotion"])
    emotion_col = emotion_data["emotion"]

    angry = len(emotion_data[emotion_col == 0])
    disgust = len(emotion_data[emotion_col == 1])
    fear = len(emotion_data[emotion_col == 2])
    happy = len(emotion_data[emotion_col == 3])
    sad = len(emotion_data[emotion_col == 4])
    surprise = len(emotion_data[emotion_col == 5])
    neutral = len(emotion_data[emotion_col == 6])

    emotion_indexes = [0, 1, 2, 3, 4, 5, 6]
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_counts = [angry, disgust, fear, happy, sad, surprise, neutral]
    max = np.array(emotion_counts).max()
    emotion_weights = [max / count for count in emotion_counts]

    df_dict = {}

    df_dict["emotion"] = emotion_indexes
    df_dict["name"] = emotion_names
    df_dict["count"] = emotion_counts
    df_dict["weight"] = emotion_weights

    weights_df = pd.DataFrame(df_dict)

    with open(name, mode='w') as csv_file:
        weights_df.to_csv(csv_file, index=False, line_terminator='\n')


getWeights(WEIGHTSPATH)