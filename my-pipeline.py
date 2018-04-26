# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:53:04 2018

@author: scott
"""

import luigi
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import pickle

# global variables
DEV_FILENAME = "train.csv"
DEV_ROOT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dev')) + "/" # directory where development data lives
PROD_ROOT = root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'prod')) + "/" # directory where production data lives
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Pclass", "Sex", "Embarked"]
DROP_COLS = ["Name", "Ticket", "Cabin"]
TARGET_COL = "Survived"
ID_COL = "PassengerId"


class InputDataDev(luigi.ExternalTask):
    """
    Returns the development data file
    """

    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    
    def output(self):
        return luigi.LocalTarget(DEV_ROOT + self.dev_filename)
    

class InputDataProd(luigi.ExternalTask):
    """
    Returns the production data file
    """

    prod_filename = luigi.Parameter()
    
    def output(self):
        return luigi.LocalTarget(PROD_ROOT + self.prod_filename)


class FitNumericImputer(luigi.Task):
    """
    Fits imputer to numeric development data 
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    
    def requires(self):
        return InputDataDev(self.dev_filename)
    
    def run(self):
        df = pd.read_csv(self.input().path)
        imputer = Imputer(strategy="median").fit(df[NUM_COLS])
        with open(self.output().path, 'wb') as f:        
            pickle.dump(imputer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(DEV_ROOT + "numeric_imputer.pickle")
    
    
class CleanData(luigi.Task):
    """
    Cleans data
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    prod_filename = luigi.Parameter(default="")
       
    def requires(self):
        if self.prod_filename == "":
            return [InputDataDev(self.dev_filename), 
                    FitNumericImputer(self.dev_filename)]
        else:
            return [InputDataProd(self.prod_filename), 
                    FitNumericImputer(self.dev_filename)]
               
    def run(self):
        df = pd.read_csv(self.input()[0].path)
        with open(self.input()[1].path, 'rb') as f:
            imputer = pickle.load(f)
        df[NUM_COLS] = imputer.transform(df[NUM_COLS]) 
        df[CAT_COLS] = df[CAT_COLS].fillna("null")
        df.drop(DROP_COLS, axis=1, inplace=True)
        df.to_csv(self.output().path, index=False) 

    def output(self):
        if self.prod_filename == "":
            return luigi.LocalTarget(DEV_ROOT + "clean.csv")  
        else:
            return luigi.LocalTarget(PROD_ROOT + "clean.csv")
        
        
class FitLabelBinarizers(luigi.Task):
    """
    Fits label binarizers to categorical development data
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    
    def requires(self):
        return CleanData(self.dev_filename)
    
    def run(self):
        df = pd.read_csv(self.input().path)        
        binarizers = {}
        for col in CAT_COLS:
            binarizers[col] = LabelBinarizer().fit(df[col])            
        with open(self.output().path, 'wb') as f:        
            pickle.dump(binarizers, f, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(DEV_ROOT + "label_binarizers.pickle")   
        

class BinarizeCategoricalData(luigi.Task):
    """
    Binarizes categorical variables
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    prod_filename = luigi.Parameter(default="")
    
    def requires(self):
        if self.prod_filename == "":
            return [CleanData(self.dev_filename), 
                    FitLabelBinarizers(self.dev_filename)]
        else:
            return [CleanData(self.dev_filename, self.prod_filename), 
                    FitLabelBinarizers(self.dev_filename)]
    
    def run(self):
        df = pd.read_csv(self.input()[0].path)  
        with open(self.input()[1].path, 'rb') as f:
            binarizers = pickle.load(f)
        for col in CAT_COLS:
            bin_data = binarizers[col].transform(df[col])
            labels = binarizers[col].classes_
            if (bin_data.shape[1] == 1) & (len(labels) == 2):
                bin_data = np.hstack((bin_data, 1 - bin_data))
            for i, lab in enumerate(labels):
                df[str(col) + "_" + str(lab)] = bin_data[:, i]
            df.drop(col, axis=1, inplace=True)
            df.to_csv(self.output().path, index=False) 

    def output(self):
        if self.prod_filename == "":
            return luigi.LocalTarget(DEV_ROOT + "clean_binarized.csv")  
        else:
            return luigi.LocalTarget(PROD_ROOT + "clean_binarized.csv")
        
        
class FitScaler(luigi.Task):
    """
    Fits scaler to numeric development data
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    
    def requires(self):
        return BinarizeCategoricalData(self.dev_filename)
    
    def run(self):
        df = pd.read_csv(self.input().path)
        scaler = StandardScaler().fit(df[NUM_COLS])
        with open(self.output().path, 'wb') as f:        
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(DEV_ROOT + "scaler.pickle")  
    

class ScaleData(luigi.Task):
    """
    Scales data
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    prod_filename = luigi.Parameter(default="")
       
    def requires(self):
        if self.prod_filename == "":
            return [BinarizeCategoricalData(self.dev_filename), 
                    FitScaler(self.dev_filename)]
        else:
            return [BinarizeCategoricalData(self.dev_filename, self.prod_filename), 
                    FitScaler(self.dev_filename)]
               
    def run(self):
        df = pd.read_csv(self.input()[0].path)
        with open(self.input()[1].path, 'rb') as f:
            scaler = pickle.load(f)
        df[NUM_COLS] = scaler.transform(df[NUM_COLS]) 
        df.to_csv(self.output().path, index=False) 

    def output(self):
        if self.prod_filename == "":
            return luigi.LocalTarget(DEV_ROOT + "clean_binarized_standardised.csv")  
        else:
            return luigi.LocalTarget(PROD_ROOT + "clean_binarized_standardised.csv")
        

class TrainModel(luigi.Task):
    """
    Trains logistic regression model on development data
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    
    def requires(self):
        return ScaleData(self.dev_filename)
    
    def run(self):
        df = pd.read_csv(self.input().path)
        X = df.drop([TARGET_COL, ID_COL], axis=1)
        y = df[TARGET_COL]
        model = LogisticRegression().fit(X, y)
        with open(self.output().path, 'wb') as f:        
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(DEV_ROOT + "model.pickle") 
    
    
class PredictTarget(luigi.Task):
    """
    Predicts target variable using logistic regression model
    """
    
    dev_filename = luigi.Parameter(default=DEV_FILENAME)
    prod_filename = luigi.Parameter(default="")
       
    def requires(self):
        if self.prod_filename == "":
            return [ScaleData(self.dev_filename), 
                    TrainModel(self.dev_filename)]
        else:
            return [ScaleData(self.dev_filename, self.prod_filename), 
                    TrainModel(self.dev_filename)]
               
    def run(self):
        df = pd.read_csv(self.input()[0].path)
        if TARGET_COL in df.columns.tolist():
            X = df.drop([TARGET_COL, ID_COL], axis=1)
        else:
            X = df.drop(ID_COL, axis=1)
        ids = df[ID_COL]
        with open(self.input()[1].path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        predictions = pd.DataFrame({ID_COL: ids, TARGET_COL: y_pred})
        predictions.to_csv(self.output().path, index=False) 

    def output(self):
        if self.prod_filename == "":
            return luigi.LocalTarget(DEV_ROOT + "predictions.csv")  
        else:
            return luigi.LocalTarget(PROD_ROOT + "predictions.csv")
        
    
if __name__ == '__main__':
    luigi.run()
        
        

    
    