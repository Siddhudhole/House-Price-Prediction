import os 
import sys 
import pandas as np 
import numpy as np 
from utils import model_evalution 
from src.Logger import logging 
from src.Exception import CustomException 
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor 
from sklearn.svm import SVR 
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts','regressor.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig() 

    def Trainer(self,train_arr,test_arr):
        try :
            logging.info('data spliting is starting')
            x_train,x_test,y_train,y_test =(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            logging.info('data spliting is completed') 
            logging.info('model training is starting') 
            models = {
                'ElasticNet':ElasticNet(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'SVR':SVR()}
            report = model_evalution(x_train,x_test,y_train,y_test,models) 
            logging.info('model training is completed') 
            return report 

        except Exception as e:
            logging.error(e)
            raise CustomException(e) 