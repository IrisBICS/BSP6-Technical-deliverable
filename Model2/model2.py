import pandas as pd
import numpy as np
from modelSuperclass import ModelSuperclass

class Model2(ModelSuperclass):

    def __init__(self, save_path, data_path, name="model1", verbose=True):

        super().__init__(save_path, data_path, name, verbose)

    def createArchitecture(self):

        if self.verbose:
            print("\nInitializing model architecture...")

        # TODO: Create model, add layers etc.

        if self.verbose:
            print("Finished initializing model architecture.")

    def loadData(self, path):

        if self.verbose:
            print("\nLoading data from", path, "...")

        # TODO: Implement data loading from csv

        if self.verbose:
            print("Extracted", len(self.x_training_data), "data entries for training.")
            print("Extracted", len(self.x_validation_data), "data entries for validation.")
            print("Extracted", len(self.x_testing_data), "data entries for testing.")
            print("Finished loading data.")

    def prepareModel(self, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        self.callbacks = []
        # TODO: Define callbacks