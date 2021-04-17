import pandas as pd
import numpy as np
from Models.NNModel import NNModel
#import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


class NN1(NNModel):

    def __init__(self, save_path, data_path, name="model1", labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'), verbose=True):

        super().__init__(save_path, data_path, name, labels, verbose)

    def createArchitecture(self):

        if self.verbose:
            print("\nInitializing model architecture...")

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
        base_model.trainable = True

        X = base_model.output
        X = Flatten()(X)
        X = Dense(64, kernel_initializer='he_uniform')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        output = Dense(7, activation='softmax')(X)

        self.model = Model(inputs=base_model.input, outputs=output)

        if self.verbose:
            print(self.model.summary())
            print("Finished initializing model architecture.")

    def loadData(self, path):

        if self.verbose:
            print("\nLoading data from", path, "...")

        # Extracting data from csv
        emotion_data = pd.read_csv(path, sep=r'\s*,\s*', engine='python')
        emotion_data.drop(columns=['Usage'])  # I am partitioning the data usage myself
        emotion_data = emotion_data.sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffling data, seeded

        all_data = []
        all_labels = []

        for _, row in emotion_data.iterrows():
            k = row['pixels'].split(" ")
            k = [float(i) / 255 for i in k]
            all_data.append(np.array(k))
            all_labels.append(int(row['emotion']))

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)

        all_data = all_data.reshape((all_data.shape[0], 48, 48))
        all_data = np.stack((all_data, all_data, all_data), axis=-1)

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

        #plt.imshow(self.x_training_data[10000])  # Plot one of the face images to see it
        #plt.show()

    def prepareModel(self, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []
        self.callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto'))

    def train(self, epochs=10, batch_size=32, learning_rate=0.0001):

        super().train(epochs, batch_size, learning_rate)