
# standard imports
import datetime
import os
from os.path import abspath, dirname, pardir, join

# third party imports
import pandas as pd 


class Submission:
    def __init__(self, results):
        self.df = self.create_df(results)

    def create_df(self, results):
        df = pd.concat(
            [
                pd.Series(range(1, 28001), name="ImageId"), 
                pd.Series(results)
            ], 
            axis=1
        )
        return df

    def save(self, directory="submissions"):
        dirpath = self.get_dirpath(directory)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_mnist_cnn.csv"
        filepath = os.path.join(dirpath, filename)
        self.df.to_csv(filepath, index=False)

    def get_dirpath(self, directory):
        parent_dirpath = abspath(join(dirname(__file__), pardir))
        dirpath = join(parent_dirpath, directory)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath
