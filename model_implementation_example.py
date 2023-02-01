import pandas as pd
from sklearn.model_selection import train_test_split

from model_class import BaseModelClass

from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearModel(BaseModelClass):
    
    #model = None
    def __init__(self, raw_data, target_col='corona_result'):
        """Initialise base class with raw training and test datasets

        Args:
            x_train (pd.DataFrame): x training data
            x_test (pd.DataFrame): x test data
            y_train (pd.Series): y training set
            y_test (pd.Series): y test set
        """
        print(f"Initialising model class of type : {self.__class__.__name__}")
        self.raw_data = raw_data
        self.target_col = target_col
    

    def preprocess_data(self):
        """function to preprocess raw data ready for model fitting
               
        return:
            pd.DataFrame
        """
        print("Preprocessing data")
        
        X = self.raw_data[['cough', 'fever', 'sore_throat', 'shortness_of_breath',
       'head_ache']]
        
        y = self.raw_data[self.target_col].map({'negative':0,
                                                'positive':1,
                                                'other':-1})
        
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        
        
        print("Setting train and test dfs")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test       #split data into x_train/test and extract target column
        
        
    

    def fit(self, X, y):
        
        print("initialising models")
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        
        print("Model fitting successful")


    def predict(self, X):
        print(f"predicting with data of shape {X.shape}")
        return self.model.predict(X)
    
class LogModel(BaseModelClass):
    
    #model = None
    def __init__(self, raw_data, target_col='corona_result'):
        """Initialise base class with raw training and test datasets

        Args:
            x_train (pd.DataFrame): x training data
            x_test (pd.DataFrame): x test data
            y_train (pd.Series): y training set
            y_test (pd.Series): y test set
        """
        print(f"Initialising model class of type : {self.__class__.__name__}")
        self.raw_data = raw_data
        self.target_col = target_col
    

    def preprocess_data(self):
        """function to preprocess raw data ready for model fitting
               
        return:
            pd.DataFrame
        """
        print("Preprocessing data")
        
        X = self.raw_data[['cough', 'fever', 'sore_throat', 'shortness_of_breath',
       'head_ache', 'age_60_and_above']]
        
        X['age_60_and_above'] = X['age_60_and_above'].map({
                                                            'No':0, 
                                                            'Yes':1
                                                            })
        nan_mask = X.age_60_and_above.isna()
        
        y = self.raw_data[self.target_col].map({'negative':0,
                                                'positive':1,
                                                'other':-1})
        
        X_train, X_test, y_train, y_test = train_test_split(X[~nan_mask],y[~nan_mask])
        
        
        print("Setting train and test dfs")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test       #split data into x_train/test and extract target column
        
        
    

    def fit(self, X, y):
        
        print("initialising models")
        self.model = LogisticRegression()
        self.model.fit(X, y)
        
        
        print("Model fitting successful")


    def predict(self, X):
        print(f"predicting with data of shape {X.shape}")
        return self.model.predict_proba(X)[:,0]
    



if __name__=='__main__':
    def score_error(predictions, expected):
        return abs(predictions - expected).mean()
    
    
    df = pd.read_csv("./data/corona_tested_individuals_ver_0083.english.csv.zip")
    
    class_object = LinearModel(df)# LogModel(df)# 
    class_object.preprocess_data()
    class_object.fit(class_object.X_train, class_object.y_train)
    predictions = class_object.predict(class_object.X_test)
    score_error(predictions, class_object.y_test)
