from Models.NNModel import NNModel
from NN1.NN1 import NN1
from NN2.NN2 import NN2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


class FinalNN(NNModel):

    def __init__(self, save_path, images_path, landmarks_path, NNImages_path, NNLandmarks_path, NNImages_name,
                 NNLandmarks_name, name="FinalNN", labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'),
                 seed=0, verbose=True):

        if verbose:
            print("\nCreating object", name)

        self.images_path = images_path
        self.landmarks_path = landmarks_path
        self.NNImages_path = NNImages_path
        self.NNLandmarks_path = NNLandmarks_path
        self.NNImages_name = NNImages_name
        self.NNLandmarks_name = NNLandmarks_name

        if verbose:
            print("\nInstantiating submodels...")

        self.NNImages = NN1(self.NNImages_path, self.images_path, name=self.NNImages_name, seed=seed)
        self.NNLandmarks = NN2(self.NNLandmarks_path, self.landmarks_path, name=self.NNLandmarks_name, seed=seed)

        self.NNImages.loadModel()
        self.NNLandmarks.loadModel()

        self.NNImages.model.trainable = False
        self.NNLandmarks.model.trainable = False

        if verbose:
            print("\nFinished instantiating submodels.")

        super().__init__(save_path, None, name, labels, seed, verbose)

        if self.verbose:
            print("\n" + self.name, "object created successfully!")

    def createArchitecture(self):

        if self.verbose:
            print("\nInitializing model architecture...")

        for layer in self.NNImages.model.layers:
            layer._name = "image_" + layer._name
        for layer in self.NNLandmarks.model.layers:
            layer._name = "landmarks_" + layer._name

        combined_out = concatenate([self.NNImages.model.output, self.NNLandmarks.model.output])
        out = Dense(7, activation='softmax')(combined_out)

        self.model = Model(inputs=[self.NNImages.model.input, self.NNLandmarks.model.input], outputs=out)

        if self.verbose:
            print(self.model.summary())
            print("Finished initializing model architecture.")

    def loadData(self, path, seed=0):  # Parameter "path" not needed here, but implemented in superclass, so I need to keep it

        self.x_training_data = [self.NNImages.x_training_data, self.NNLandmarks.x_training_data]
        self.y_training_data = self.NNLandmarks.y_training_data  # Equivalent to self.NNLandmarks.y_training_data, thanks to seed
        self.x_validation_data = [self.NNImages.x_validation_data, self.NNLandmarks.x_validation_data]
        self.y_validation_data = self.NNLandmarks.y_validation_data
        self.x_testing_data = [self.NNImages.x_testing_data, self.NNLandmarks.x_testing_data]
        self.y_testing_data = self.NNLandmarks.y_testing_data

    def prepareModel(self, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []

    def train(self, epochs=10, batch_size=32, learning_rate=0.001):

        super().train(epochs, batch_size, learning_rate)