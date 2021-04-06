import os
import csv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


class Model1:

    def __init__(self, save_path, data_path, name="model1", verbose=True):

        if verbose:
            print("\nCreating model object...")

        self.save_path = save_path
        self.name = name
        self.verbose = verbose

        self.__loadData(data_path)

        if self.verbose:
            print("\nModel object created successfully!")

    def __createArchitecture(self):

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
            print("Finished initializing model architecture.")

    def __loadData(self, path):

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

    def __prepareModel(self, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []
        self.callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto'))

    def train(self, epochs=10, batch_size=32, learning_rate=0.0001):

        self.__createArchitecture()
        self.__prepareModel(epochs, batch_size)

        if self.verbose:
            print("\nStarting training...")

        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        self.history = self.model.fit(self.x_training_data, self.y_training_data, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.x_validation_data, self.y_validation_data), callbacks=self.callbacks)

        if self.verbose:
            print("Finished training.")

        self.__saveModel()

    def __saveModel(self):

        if self.verbose:
            print("\nSaving model to disk...")

        # Save architecture
        model_json = self.model.to_json()
        with open(os.path.join(self.save_path, self.name + ".json"), mode="w") as json_file:
            json_file.write(model_json)

        # Save weights
        self.model.save_weights(os.path.join(self.save_path, self.name + ".h5"))

        # Save history
        history_df = pd.DataFrame.from_dict(self.history.history)
        with open(os.path.join(self.save_path, self.name + ".csv"), mode='w') as csv_file:
            history_df.to_csv(csv_file)

        if self.verbose:
            print("Finished saving model to disk.")

    def loadModel(self):

        if self.verbose:
            print("\nLoading model from location", self.save_path, "...")

        self.model = model_from_json(open(os.path.join(self.save_path, self.name + ".json"), "r").read())
        self.model.load_weights(os.path.join(self.save_path, self.name + '.h5'))

        historyCols = []
        with open(os.path.join(self.save_path, self.name + ".csv"), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if historyCols:
                    for i, value in enumerate(row):
                        historyCols[i].append(float(value))
                else:
                    historyCols = [[value] for value in row]
        self.history = {c[0]: c[1:] for c in historyCols}

        self.optimizer = Adam(learning_rate=self.history["lr"][-1])  # Take last lr value as lr
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        if self.verbose:
            print("Finished loading model.")

    def __evaluate(self, x, y):

        loss_and_metrics = self.model.evaluate(x, y)
        print("Loss: ", loss_and_metrics[0], "\nAccuracy: ", loss_and_metrics[1])

    def evaluateOnValid(self):

        if not self.model:
            print("\nError: Please load or train the model first!")
            return

        if self.verbose:
            print("\nEvaluating the model on the validation data...")

        self.__evaluate(self.x_validation_data, self.y_validation_data)

        if self.verbose:
            print("Finished evaluating the model on the validation data.")

    def evaluateOnTest(self):

        if not self.model:
            print("\nError: Please load or train the model first!")
            return

        if self.verbose:
            print("\nEvaluating the model on the unseen testing data...")

        self.__evaluate(self.x_testing_data, self.y_testing_data)

        if self.verbose:
            print("Finished evaluating the model on the unseen testing data.")