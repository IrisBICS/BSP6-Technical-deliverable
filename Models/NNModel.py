import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json
from sklearn.metrics import classification_report, confusion_matrix


class NNModel:

    def __init__(self, save_path, data_path, weights_path, name, labels, seed=0, verbose=True):

        if verbose:
            print("\nCreating object", name)

        self.save_path = save_path
        self.name = name
        self.labels = labels
        self.verbose = verbose

        self.loadData(data_path, seed=seed)
        self.loadWeights(weights_path)

        if self.verbose:
            print("\n" + self.name, "object created successfully!")

    def createArchitecture(self):  # Override implementation in subclass

        if self.verbose:
            print("\nWarning: You are calling the superclass method implementation of createArchitecture, but you should override this implementation in your subclass!")
            print("\nInitializing model architecture...")

        self.model = Model()

        if self.verbose:
            print("Finished initializing model architecture.")

    def loadWeights(self, path):

        if self.verbose:
            print("\nLoading weights from", path, "...")

        weights_data = pd.read_csv(path, sep=r'\s*,\s*', engine='python')
        self.weights = weights_data["weight"].to_dict()

        if self.verbose:
            print("Finished loading weights.")

    def loadData(self, path, seed=0):  # Override implementation in subclass

        if self.verbose:
            print("\nWarning: You are calling the superclass method implementation of loadData, but you should override this implementation in your subclass!")
            print("\nLoading data from", path, "...")

        self.x_training_data = []  # Data to train the model
        self.y_training_data = []
        self.x_validation_data = []  # Data for model validation after each epoch
        self.y_validation_data = []
        self.x_testing_data = []  # Data for testing the model (unseen data from the same dataset)
        self.y_testing_data = []

        if self.verbose:
            print("Extracted", len(self.x_training_data), "data entries for training.")
            print("Extracted", len(self.x_validation_data), "data entries for validation.")
            print("Extracted", len(self.x_testing_data), "data entries for testing.")
            print("Finished loading data.")

    def prepareModel(self, epochs, batch_size):  # Override implementation in subclass

        print("\nWarning: You are calling the superclass method implementation of prepareModel, but you should override this implementation in your subclass!")

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []

    def train(self, epochs, batch_size, learning_rate):

        self.createArchitecture()
        self.prepareModel(epochs, batch_size)

        if self.verbose:
            print("\nStarting training...")

        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])

        self.history = self.model.fit(self.x_training_data, self.y_training_data, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(self.x_validation_data, self.y_validation_data), callbacks=self.callbacks, class_weight=self.weights)

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
            history_df.to_csv(csv_file, index=True, line_terminator='\n')

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

        try:
            self.optimizer = Adam(learning_rate=self.history["lr"][-1])  # Take last lr value as lr
        except:
            self.optimizer = Adam(learning_rate=10**(-8))
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

    def __confusionMatrix(self, x, y, dtype, percentage=True, normalize=False, save=True, show=True, report=False):

        y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)

        fig, ax = plt.subplots(figsize=(7, 7))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(self.labels)))

        if percentage:
            if normalize:
                conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)[:,None]
            else:
                conf_matrix = conf_matrix / np.sum(conf_matrix, axis=None)
            fmt = "0.2%"
        else:
            fmt = "1"

        sns.heatmap(conf_matrix, annot=True, annot_kws={"size":12}, fmt=fmt, square=True, cmap=plt.cm.get_cmap(), cbar=False, xticklabels=self.labels, yticklabels=self.labels, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix for ' + self.name + ' ' + dtype + ' data')
        if save:
            part = "_cm_"
            if percentage:
                if normalize:
                    part += "norm_"
            else:
                part += "abs_"
            name = os.path.join(self.save_path, "plots/", self.name + part + dtype + ".png")
            plt.savefig(name)
        if show:
            plt.show()

        if report:
            print('Classification Report for', self.name, "with", dtype, "data:")
            print(classification_report(y_true, y_pred, labels=np.arange(len(self.labels)), target_names=self.labels))

    def generateConfusionMatrices(self, percentage=True, normalize=False, save=True, show=True, report=False):

        if not self.model:
            print("\nError: Please load or train the model first!")
            return

        if self.verbose:
            print("\nGenerating the confusion matrix for the training data...")

        self.__confusionMatrix(self.x_training_data, self.y_training_data, "training", percentage=percentage, normalize=normalize, save=save, show=show, report=report)

        if self.verbose:
            print("Generating the confusion matrix for the validation data...")

        self.__confusionMatrix(self.x_validation_data, self.y_validation_data, "validation", percentage=percentage, normalize=normalize, save=save, show=show, report=report)

        if self.verbose:
            print("Generating the confusion matrix for the testing data...")

        self.__confusionMatrix(self.x_testing_data, self.y_testing_data, "testing", percentage=percentage, normalize=normalize, save=save, show=show, report=report)

        if self.verbose:
            print("Finished generating the confusion matrices.")

    def generatePlots(self, accuracy=True, loss=True, learning_rate=True, save=True, show=True):

        if not self.history:
            print("\nError: Please load or train the model first!")
            return

        if self.verbose:
            print("\nGenerating the plots...")

        tr_acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']
        tr_loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = range(len(tr_acc))

        save_prefix = os.path.join(self.save_path, "plots/", self.name)

        # Accuracy
        if accuracy:
            plt.plot(epochs, tr_acc, 'r-', label='Training Accuracy')
            plt.plot(epochs, val_acc, 'b--', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy of ' + self.name)
            plt.legend(loc='best')

            if save:
                plt.savefig(save_prefix + "_accuracy.png")
            if show:
                plt.show()

        # Loss
        if loss:
            plt.plot(epochs, tr_loss, 'r-', label='Training Loss')
            plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
            plt.title('Training and Validation Loss of ' + self.name)
            plt.legend(loc='best')

            if save:
                plt.savefig(save_prefix + "_loss.png")
            if show:
                plt.show()

        # Learning rate
        if learning_rate:
            lr = self.history['lr']
            plt.plot(epochs, lr, 'r-', label='Learning rate')
            plt.title('Learning rate over training time of ' + self.name)
            plt.legend(loc='best')

            if save:
                plt.savefig(save_prefix + "_lr.png")
            if show:
                plt.show()

        if self.verbose:
            print("Finished generating the plots.")