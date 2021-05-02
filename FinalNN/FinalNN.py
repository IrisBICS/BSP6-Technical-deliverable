import os
import matplotlib.pyplot as plt
from Models.NNModel import NNModel
from NN1.NN1 import NN1
from NN2.NN2 import NN2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


class FinalNN(NNModel):

    def __init__(self, save_path, images_path, landmarks_path, weights_path, NNImages_path, NNLandmarks_path, NNImages_name,
                 NNLandmarks_name, name="FinalNN", labels=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'),
                 combine_mode="decisions", trainable=False, seed=0, verbose=True):

        if verbose:
            print("\nCreating object", name)

        self.images_path = images_path
        self.landmarks_path = landmarks_path
        self.weights_path = weights_path
        self.NNImages_path = NNImages_path
        self.NNLandmarks_path = NNLandmarks_path
        self.NNImages_name = NNImages_name
        self.NNLandmarks_name = NNLandmarks_name
        self.combine_mode = combine_mode

        if verbose:
            print("\nInstantiating submodels...")

        self.NNImages = NN1(self.NNImages_path, self.images_path, self.weights_path, name=self.NNImages_name, seed=seed)
        self.NNLandmarks = NN2(self.NNLandmarks_path, self.landmarks_path, self.weights_path, name=self.NNLandmarks_name, seed=seed)

        self.NNImages.loadModel()
        self.NNLandmarks.loadModel()

        self.NNImages.model.trainable = trainable
        self.NNLandmarks.model.trainable = trainable

        if verbose:
            print("\nFinished instantiating submodels.")

        super().__init__(save_path, None, self.weights_path, name, labels, seed, verbose)

        if self.verbose:
            print("\n" + self.name, "object created successfully!")

    def createArchitecture(self):

        if self.verbose:
            print("\nInitializing model architecture...")

        for layer in self.NNImages.model.layers:
            layer._name = "image_" + layer._name
        for layer in self.NNLandmarks.model.layers:
            layer._name = "landmarks_" + layer._name

        if self.combine_mode == "features":  # Model will process the features extracted by each model to make a final decision
            combined_out = concatenate([self.NNImages.model.layers[-2].output, self.NNLandmarks.model.layers[-2].output])
            combined_out = Dense(64)(combined_out)
            combined_out = Dense(16)(combined_out)
        else:  # By default, the model combines the decisions
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

    def train(self, epochs=16, batch_size=32, learning_rate=0.00005):

        super().train(epochs, batch_size, learning_rate)

    def generatePlots(self, accuracy=True, loss=True, learning_rate=True, save=True, show=True):

        if not self.history:
            print("\nError: Please load or train the model first!")
            return

        super().generatePlots(learning_rate=False)

        if self.verbose:
            print("\nGenerating comparison plots to individual models...")

        save_prefix = os.path.join(self.save_path, "plots/", self.name)

        val_acc = self.history['val_accuracy']
        img_val_acc = self.NNImages.history['val_accuracy']
        lmk_val_acc = self.NNLandmarks.history['val_accuracy']
        val_loss = self.history['val_loss']
        img_val_loss = self.NNImages.history['val_loss']
        lmk_val_loss = self.NNLandmarks.history['val_loss']

        epochs = range(len(val_acc))
        epochs_img = range(len(img_val_acc))
        epochs_lmk = range(len(lmk_val_acc))

        if accuracy:
            plt.plot(epochs, val_acc, 'r-', label='Final model')
            plt.plot(epochs_img, img_val_acc, 'b-', label='Images submodel')
            plt.plot(epochs_lmk, lmk_val_acc, 'g-', label='Landmarks submodel')
            plt.title('Validation Accuracy of final model and its submodels')
            plt.legend(loc='best')

            if save:
                plt.savefig(save_prefix + "_accuracy_comparison.png")
            if show:
                plt.show()

        if loss:
            plt.plot(epochs, val_loss, 'r-', label='Final model')
            plt.plot(epochs_img, img_val_loss, 'b-', label='Images submodel')
            plt.plot(epochs_lmk, lmk_val_loss, 'g-', label='Landmarks submodel')
            plt.title('Validation Loss of final model and its submodels')
            plt.legend(loc='best')

            if save:
                plt.savefig(save_prefix + "_loss_comparison.png")
            if show:
                plt.show()

        if self.verbose:
            print("Finished generating the comparison plots.")