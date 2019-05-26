
# standard imports
import os
from os.path import abspath, dirname, join, pardir
import inspect
import random

# third party imports
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# local imports
from helpers import timer

    
PARENT_DIRPATH = abspath(join(dirname(__file__), pardir))    
DATASETS_DIRPATH = join(PARENT_DIRPATH, "datasets")


class DataSet:
    def __init__(self, filename=None, dirpath=DATASETS_DIRPATH):
        if not os.path.exists(dirpath):
            raise AttributeError(f"'{dirpath} does not exists'")

        self.dirpath = dirpath
        self.filename = filename

        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.labels_type = "digits"

    def __repr__(self):
        dataset_name = str(inspect.stack()[1][4]).split('(')[1].split(')')[0]
        object_repr = f"\n{dataset_name.upper()} dataset:\n"
        object_repr += f"  source   | {self.filename}\n"
        object_repr += f"  X.shape  | {self.X.shape}\n"
        object_repr += f"  Y.shape  | {self.Y.shape}"
        return object_repr

    @timer
    def download(self, nrows=None):
        """Download csv file to pandas dataframe"""
        
        if nrows:
            print(f"\ndownload {nrows} first rows of '{self.filename}' dataset ...")
        else:
            print(f"\ndownload '{self.filename}' dataset ...")

        filepath = join(self.dirpath, self.filename)
        df = pd.read_csv(filepath, nrows=nrows)

        self.df = df
        return df

    def set_X(self):
        """Set X if dataframe does not contain Y data."""
        self.X = self.df

    def split_X_Y(self):
        """Split X and Y values from the dataframe."""
        print("split X and Y")
        self.X = self.df.drop(labels=["label"], axis=1)
        self.Y = self.df["label"]

    def normalize(self, max_value=255.):
        """Normalize grayscale values. (CNN converge faster on [0,1] data)"""
        print("normalize X grayscale values")
        self.X = self.X / max_value

    def reshape(self, matrix_shape=(-1, 28, 28, 1)):
        """Reshape 1D vector to 3D matrices"""
        print("reshape X 1D vectors to matrices")
        self.X = self.X.values.reshape(matrix_shape)

    def convert_digits_to_one_hot_vectors(self, num_classes=10):
        """
        Encode digits labels to one hot vectors.

        Example
        -------
            `4 -> [0,0,0,1,0,0,0,0,0,0,0]`
        """
        print("convert digit labels to one hot vectors")
        assert self.labels_type == "digits"
        self.Y = to_categorical(self.Y, num_classes=num_classes)
        self.labels_type = "one_hot"

    def convert_one_hot_vectors_to_digits(self):
        assert self.labels_type == "one_hot"
        self.Y = np.argmax(self.Y, axis=1)
        self.labels_type = "digits"

    def extract_validation(self, size=0.1, random_seed=2):
        """split train and validation dataset"""
        print("extract validation dataset from train dataset")
        validation = DataSet()
        validation.filename = self.filename
        split = train_test_split(self.X, self.Y, test_size=size, random_state=random_seed)
        self.X = split[0]
        validation.X = split[1]
        self.Y = split[2]
        validation.Y = split[3]

        return validation

    def plot_digit(self, index=None):
        """
        Plot digit

        Parameters 
        -------
        index : int, optionnal
            Digit index, defaut to random index.
        """
        print("plot digit")
        if not index:
            index = int(random.random() * self.X.shape[0])

        undo_labeling = False
        if self.labels_type == "one_hot":
            undo_labeling = True
            self.convert_one_hot_vectors_to_digits()


        plt.title(self.Y[index], weight='bold', fontsize=20)
        plt.imshow(self.X[index][:, :, 0], cmap='binary')
        plt.show()

        if undo_labeling:
            self.convert_digits_to_one_hot_vectors()
