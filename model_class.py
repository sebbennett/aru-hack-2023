import pandas as pd
from abc import ABC, abstractmethod

class BaseModelClass(ABC):
    
    @abstractmethod
    def __init__(self, raw_data:pd.DataFrame, target_col:str):
        """Initialise base class with raw training and test datasets

        Args:
            x_train (pd.DataFrame): x training data
            x_test (pd.DataFrame): x test data
            y_train (pd.Series): y training set
            y_test (pd.Series): y test set
        """
        pass
    
    @abstractmethod
    def preprocess_data(self):
        """function to preprocess raw data ready for model fitting
               
        return:
            pd.DataFrame
        """
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    


    



if __name__ == '__main__':
    pass