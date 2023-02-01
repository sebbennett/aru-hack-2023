import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model_class import BaseModelClass
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import PrecisionRecallDisplay



class LogHolidayModel(BaseModelClass):
    
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
        X = self.raw_data.copy()
        ISRAEL_PUBLIC_HOLIDAYS = pd.read_html('./israel_holidays.html')[0]
        ISRAEL_PUBLIC_HOLIDAYS['full_date'] = ISRAEL_PUBLIC_HOLIDAYS['Date'] + ' 2020'
        ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'] = pd.to_datetime(ISRAEL_PUBLIC_HOLIDAYS['full_date'], format='%b %d %Y')

        name_date = ISRAEL_PUBLIC_HOLIDAYS[['Holiday Name','full_date_datetime']].set_index('Holiday Name').T
        values = name_date.values
        holiday_values = np.array([values[0] for x in X.index])

        holiday_df = pd.DataFrame(holiday_values, columns=name_date.columns.to_list())
        joined_df = pd.concat([X, holiday_df], axis=1)

        joined_df['test_date'] = pd.to_datetime(joined_df['test_date'])

        for col in name_date.columns:
            print(col)
            try:
                joined_df[col + '_diff'] = (joined_df['test_date'] - pd.to_datetime(joined_df[col])) / np.timedelta64(1,'D')
            except:
                pass
            
        diff_cols = [col for col in joined_df.columns if col.endswith('_diff')]


        joined_df['closest_event'] = joined_df[diff_cols].apply(abs).idxmin(axis='columns')

        X_all = pd.concat([X,pd.get_dummies(joined_df['closest_event'])], axis=1)


        X = X_all[['test_date','cough', 'fever', 'sore_throat', 'shortness_of_breath',
            'head_ache','age_60_and_above', 'gender', 'corona_result']+list(joined_df['closest_event'].unique())]


        X['age_60_and_above'] = X['age_60_and_above'].map({'No':1, 'Yes':2}).fillna(-1)
        X['gender'] = X['gender'].map({'male':1, 'female':2}).fillna(-1)
        
        X= X[X[self.target_col] != 'other']
        
        y = X[self.target_col].map({'negative':0,
                                                'positive':1,
                                                'other':-1})
        
        self.X = X
        self.y = y
        #X_train, X_test, y_train, y_test = train_test_split(X.drop(self.target_col, axis=1),y)
        X_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))].drop(['test_date','corona_result'], axis = 1)
        y_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))]['corona_result']
        X_test = X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))].drop(['test_date','corona_result'], axis = 1)
        y_test =  X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))]['corona_result']#.drop(['corona_result'], axis = 1)
      
        
        print("Setting train and test dfs")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test       #split data into x_train/test and extract target column
        
        
    

    def fit(self, X, y):
        
        print("initialising models")
        #self.model = LogisticRegression()
        self.model.fit(X, y)
        
        
        print("Model fitting successful")


    def predict(self, X):
        print(f"predicting with data of shape {X.shape}")
        return self.model.predict(X)
    
class PaperModel(BaseModelClass):
    
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
        X = self.raw_data.copy()
        
        num_columns = ['cough', 'fever', 'sore_throat', 'shortness_of_breath',
       'head_ache']
        for i in num_columns:
            X = X[X[i] != 'None']
            print (pd.unique(X[i]))
        X[num_columns] = X[num_columns].apply(pd.to_numeric)
        X[num_columns].dtypes
        
        
        cat_columns = [ 'age_60_and_above', 'gender',
       'test_indication']
        for i in cat_columns:
            X = X[X[i] != 'None']
            print (pd.unique(X[i]))
            
        X = X[X['corona_result'] != 'other']
        

        y = X["corona_result"]
        #y = y.map({"negative":0, "positive" : 1})
        X["corona_result"] = X["corona_result"].map({"negative":0, "positive" : 1})


        X = pd.get_dummies(X, columns = cat_columns)# + num_columns)
       
        self.X = X
        
        X_train, X_test, y_train, y_test = train_test_split(X.drop(['test_date','corona_result'], axis = 1), X['corona_result'], test_size = 0.25)
    
        # X_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))].drop(['test_date','corona_result'], axis = 1)
        # y_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))]['corona_result']
        # X_test = X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))].drop(['test_date','corona_result'], axis = 1)
        # y_test =  X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))]['corona_result']#.drop(['corona_result'], axis = 1)
        print("Setting train and test dfs")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test       #split data into x_train/test and extract target column
        
        
    

    def fit(self, X, y):
        

        
        print("initialising models")

        self.model.fit(X, y)
        
        
        print("Model fitting successful")


    def predict(self, X):
        print(f"predicting with data of shape {X.shape}")
        return self.model.predict(X)

class PaperDateSplitModel(BaseModelClass):
    
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
        X = self.raw_data.copy()
        
        num_columns = ['cough', 'fever', 'sore_throat', 'shortness_of_breath',
       'head_ache']
        for i in num_columns:
            X = X[X[i] != 'None']
            print (pd.unique(X[i]))
        X[num_columns] = X[num_columns].apply(pd.to_numeric)
        X[num_columns].dtypes
        
        
        cat_columns = [ 'age_60_and_above', 'gender',
       'test_indication']
        for i in cat_columns:
            X = X[X[i] != 'None']
            print (pd.unique(X[i]))
            
        X = X[X['corona_result'] != 'other']
        

        y = X["corona_result"]
        #y = y.map({"negative":0, "positive" : 1})
        X["corona_result"] = X["corona_result"].map({"negative":0, "positive" : 1})


        X = pd.get_dummies(X, columns = cat_columns)
       
        self.X = X
        
        # X_train, X_test, y_train, y_test = train_test_split(X.drop(['test_date','corona_result'], axis = 1), X['corona_result'], test_size = 0.25)
    
        X_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))].drop(['test_date','corona_result'], axis = 1)
        y_train = X[((X['test_date'] >= '2020-03-22') & (X['test_date'] <= '2020-03-31'))]['corona_result']
        X_test = X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))].drop(['test_date','corona_result'], axis = 1)
        y_test =  X[((X['test_date'] >= '2020-04-01') & (X['test_date'] <= '2020-04-07'))]['corona_result']#.drop(['corona_result'], axis = 1)
        print("Setting train and test dfs")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test       #split data into x_train/test and extract target column
        
        
    

    def fit(self, X, y):
        

        
        print("initialising models")

        self.model.fit(X, y)
        
        
        print("Model fitting successful")


    def predict(self, X):
        print(f"predicting with data of shape {X.shape}")
        return self.model.predict(X)

def test_model(model):
    
    class_object.model = model#GaussianNB()
    class_object.fit(class_object.X_train, class_object.y_train)
    predictions = class_object.predict(class_object.X_test)


    #class_object.model.score(class_object.X_test, class_object.y_test)

    ConfusionMatrixDisplay(confusion_matrix(class_object.y_test,predictions), display_labels=class_object.model.classes_).plot()#.show()
    print(classification_report(class_object.y_test,predictions))
    

    
    fpr, tpr, _ = roc_curve(class_object.y_test, class_object.model.predict_proba(class_object.X_test)[:,1])
    roc_auc= auc(fpr, tpr)
    print(f"roc_auc = {roc_auc}")
    
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='C2',
            lw=lw, label=f'{model.__class__.__name__}')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, ls='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(visible=True)
    plt.xlabel('1-specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.legend(fontsize=12)

    plt.show()


    

    display = PrecisionRecallDisplay.from_estimator(
        model, class_object.X_test, class_object.y_test, 
    )
    _ = display.ax_.set_title("Precision-Recall curve")



df = pd.read_csv("./data/corona_tested_individuals_ver_0083.english.csv.zip")


print('='*20 + ' TRAIN_TEST' +'='*20)
class_object = PaperModel(df)#  LinearModel(df)#
class_object.preprocess_data()
test_model(LGBMClassifier(class_weight="balanced", n_jobs=-1))



print('='*20 + ' PaperDateSplitModel ' +'='*20)
class_object = PaperDateSplitModel(df)#  LinearModel(df)#
class_object.preprocess_data()
test_model(LGBMClassifier(class_weight="balanced", n_jobs=-1))


print('='*20 + ' LogHolidayModel ' +'='*20)
class_object = LogHolidayModel(df)#  LinearModel(df)#
class_object.preprocess_data()
test_model(LGBMClassifier(class_weight="balanced", n_jobs=-1))












####


# from xgboost import XGBClassifier
# from sklearn.ensemble import StackingClassifier

# #
# # Load the IRIS dataset
# #

# # Create a Randomforest classifier
# #
# forest = RandomForestClassifier(n_estimators=100,
#                                 random_state=123)
# #
# # Create a XGBoost classifier
# #
# boost = XGBClassifier(random_state=123, verbosity=0, use_label_encoder=False)
# #
# # Create a Logistic regression classifier
# #
# lgclassifier = LogisticRegression(random_state=123)
# #
# # Create a stacking classifier
# #
# estimators = [
#     ('rf', forest),
#     ('xgb', boost)
# ]
# sclf = StackingClassifier(estimators=estimators,
#     final_estimator=lgclassifier,
#     cv=10)
# #
# # Fit the random forest classifier; Print the scores
# #
# test_model(forest)

# # Fit the XGBoost classifier; Print the scores
# #
# print('='*20 + ' FOREST ' +'='*20)
# test_model(boost)
# print('='*20 + ' BOOST ' +'='*20)
# #
# # Fit the Stacking classifier; Print the scores
# #
# test_model(sclf)
# print('='*20 + ' SCLF ' +'='*20)


# df = pd.read_csv("./data/corona_tested_individuals_ver_006.english.csv.zip")


# def score_error(predictions, expected):
#         return abs(predictions - expected).mean()
    
    
# #df = pd.read_csv("./data/corona_tested_individuals_ver_0083.english.csv.zip")
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_curve, auc, roc_auc_score



# class_object = LogHolidayModel(df)#  LinearModel(df)#
# class_object.preprocess_data()

# from sklearn.ensemble import BaggingClassifier

# # Your code here...

# ensemble = BaggingClassifier(
#     base_estimator=GaussianNB(),
#     n_estimators=10,
#     random_state=5
# )
# rf = RandomForestClassifier(random_state=5)

# class_object.model = rf#GaussianNB()
# class_object.fit(class_object.X_train, class_object.y_train)
# predictions = class_object.predict(class_object.X_test)


# #class_object.model.score(class_object.X_test, class_object.y_test)

# ConfusionMatrixDisplay(confusion_matrix(class_object.y_test,predictions), display_labels=class_object.model.classes_).plot()
# print(classification_report(class_object.y_test,predictions))
