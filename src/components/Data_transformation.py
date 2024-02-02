import os 
import sys 
import pandas as pd 
import numpy as np 
from utils import save_model 
from src.Logger import logging
from src.Exception import CustomException 
from sklearn.pipeline import Pipeline 
from sklearn.compose  import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder  
from dataclasses import dataclass 



@dataclass 
class DataTransformationCofig:
    processor_file_path = os.path.join('artifacts','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.processor_config = DataTransformationCofig()

    def get_processor(self):
        try :
            logging.info("Pipeline creation are started...")
            num_cols = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
            cat_cols = ['Furnishing', 'Status', 'Transaction', 'Type']
            num_pip = Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaler',StandardScaler())]) 
            cat_pip = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OrdinalEncoder(categories='auto'))])
            logging.info("Pipeline creation are completed...") 
            logging.info("Preprocessor creation are started...")
            preprocessor = ColumnTransformer([('num',num_pip,num_cols),('cat',cat_pip,cat_cols)]) 
            logging.info("Preprocessor creation are completed...") 
            return preprocessor 
        
        except CustomException as e:
            logging.error(e)
            raise CustomException(e,sys) 
        
    def Transformer(self,train_path,test_path):
        try :
            logging.info("Transformation is starting") 
            train_df = pd.read_csv(train_path)
            target_feature = 'Price' 
            train_input_features = train_df.drop('Price',axis=1)
            train_target_features = train_df[target_feature] 
            logging.info('Preprocessor obtaining starting ')
            preprocessor = self.get_processor() 
            logging.info('Preprocessor obtained successfully')
            train_input_arr = preprocessor.fit_transform(train_input_features)
            logging.info('save preprocessor object start ')
            save_model(self.processor_config.processor_file_path,preprocessor) 
            logging.info('save preprocessor object completed') 
            train_arr = np.c_[np.array(train_input_arr),np.array(train_target_features)]
            logging.info('Training data is tranformation successful')
            test_df = pd.read_csv(test_path) 
            test_input_features = test_df.drop('Price',axis=1) 
            test_target_features = test_df[target_feature] 
            test_input_arr = preprocessor.transform(test_input_features) 
            test_arr = np.c_[np.array(test_input_arr),np.array(test_target_features)]
            logging.info('Test data is tranformation successful') 
            return train_arr,test_arr 

        except CustomException as e:
            logging.error(e)
            raise CustomException(e,sys) 
        


