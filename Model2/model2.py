import pandas as pd
import numpy as np
from modelSuperclass import ModelSuperclass
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

class Model2(ModelSuperclass):

    def __init__(self, save_path, data_path, name="model2", verbose=True):

        super().__init__(save_path, data_path, name, verbose)

    def createArchitecture(self):

        if self.verbose:
            print("\nInitializing model architecture...")

        inp = Input(shape=(68, 2))  # Takes all 68 landmarks as input

        # Mask for every single facial feature group
        jaw_inp = Lambda(lambda x: x[:, :17])(inp)
        left_brow_inp = Lambda(lambda x: x[:, 17:22])(inp)
        right_brow_inp = Lambda(lambda x: x[:, 22:27])(inp)
        nose_inp = Lambda(lambda x: x[:, 27:36])(inp)
        right_eye_inp = Lambda(lambda x: x[:, 36:42])(inp)
        left_eye_inp = Lambda(lambda x: x[:, 42:48])(inp)
        mouth_inp = Lambda(lambda x: x[:, 48:61])(inp)
        lips_inp = Lambda(lambda x: x[:, 61:])(inp)

        # Process each feature group with 2 neurons, independently from the other feature groups
        jaw_dense = Dense(2, activation="sigmoid")(jaw_inp)
        left_brow_dense = Dense(2, activation="sigmoid")(left_brow_inp)
        right_brow_dense = Dense(2, activation="sigmoid")(right_brow_inp)
        nose_dense = Dense(2, activation="sigmoid")(nose_inp)
        right_eye_dense = Dense(2, activation="sigmoid")(right_eye_inp)
        left_eye_dense = Dense(2, activation="sigmoid")(left_eye_inp)
        mouth_dense = Dense(2, activation="sigmoid")(mouth_inp)
        lips_dense = Dense(2, activation="sigmoid")(lips_inp)

        concat = concatenate(axis=1, inputs=[jaw_dense, left_brow_dense, right_brow_dense, nose_dense, right_eye_dense, left_eye_dense, mouth_dense, lips_dense])  # Layer with 8 neurons, each corresponding to one feature group
        flatten = Flatten()(concat)

        # From here onwards, architecture taken from "Facial Expression Recognition using Facial Landmark Detection and Feature Extraction via Neural Networks" (F. Kahn)
        hidden = Dense(100, activation="sigmoid")(flatten)
        hidden = Dense(500, activation="sigmoid")(hidden)
        dropout = Dropout(0.3)(hidden)
        out = Dense(7, activation="sigmoid")(dropout)

        self.model = Model(inputs=inp, outputs=out)

        if self.verbose:
            print(self.model.summary())
            print("Finished initializing model architecture.")

    def loadData(self, path):

        if self.verbose:
            print("\nLoading data from", path, "...")

        emotion_data = pd.read_csv(path, sep=r'\s*,\s*', engine='python')
        emotion_data = emotion_data.sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffling data, seeded

        all_data = []
        all_labels = []

        for _, row in emotion_data.iterrows():
            emotion = int(row["emotion"])
            all_labels.append(emotion)

            data = []
            for part in ["jaw", "right_brow", "left_brow", "nose", "right_eye", "left_eye", "mouth", "lips"]:
                data += self.__splitData(row[part])

            all_data.append(data)

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)

        all_labels = to_categorical(all_labels, num_classes=7)

        total_amount = len(all_data)
        val_amount = int(np.round(total_amount * 0.2))  # 20%
        test_amount = int(np.round(total_amount * 0.1))  # 10%
        boundary1 = total_amount - val_amount - test_amount
        boundary2 = total_amount - test_amount

        self.x_training_data = all_data[:boundary1]  # Data to train the model
        self.y_training_data = all_labels[:boundary1]
        self.x_validation_data = all_data[boundary1:boundary2]  # Data for model validation after each epoch
        self.y_validation_data = all_labels[boundary1:boundary2]
        self.x_testing_data = all_data[boundary2:]  # Data for testing the model (unseen data from the same dataset)
        self.y_testing_data = all_labels[boundary2:]

        if self.verbose:
            print("Extracted", len(self.x_training_data), "data entries for training.")
            print("Extracted", len(self.x_validation_data), "data entries for validation.")
            print("Extracted", len(self.x_testing_data), "data entries for testing.")
            print("Finished loading data.")

    def __splitData(self, data):

        points = data.split(" ")
        xy_split = [point.split(".") for point in points]

        float_xy = []
        for [x, y] in xy_split:
            float_xy.append([float(x), float(y)])

        return float_xy

    def prepareModel(self, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []
        # TODO: Define callbacks if any is needed - starting with no callback

    def train(self, epochs=25, batch_size=32, learning_rate=0.005):

        super().train(epochs, batch_size, learning_rate)