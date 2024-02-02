import os 
import sys 
import pandas as pd 
from dataclasses import dataclass 
from src.Logger import logging 
from src.Exception import CustomException 
from sklearn.model_selection import train_test_split 

@dataclass 
class DataIngestionConfig:
    train_data_path =  os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv') 


class DataIngestion:
    def __init__(self):
        self.files_path = DataIngestionConfig()

    def load_data(self):
        try:
            logging.info("Loading data starting")
            df = pd.read_csv('notebook\data\MagicBricks.csv')
            logging.info("loaded data successfully")
            train_set,test_test = train_test_split(df,test_size=0.20,random_state=0)
            os.makedirs(os.path.dirname(self.files_path.train_data_path),exist_ok=True)
            df.to_csv(self.files_path.raw_data_path)
            logging.info('Data saved to csv file successfully')
            train_set.to_csv(self.files_path.train_data_path)
            logging.info('Train set saved to csv file successfully')
            test_test.to_csv(self.files_path.test_data_path)
            logging.info('Test set saved to csv file successfully') 
            return self.files_path.train_data_path,self.files_path.test_data_path 
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
    


