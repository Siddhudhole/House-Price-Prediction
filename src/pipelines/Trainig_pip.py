import os 
import sys 
from src.Logger import logging 
from src.Exception import CustomException 
from src.components.Data_ingestion import DataIngestionConfig 
from src.components.Data_ingestion import DataIngestion 
from src.components.Data_transformation import DataTransformationCofig  
from src.components.Data_transformation import DataTransformation  


class Trainig_pipeline :
    try :
        logging.info("Training pipeline starting") 
        logging.info('data loading module starting')
        data_ingestion = DataIngestion()
        train_path,test_path = data_ingestion.load_data() 
        logging.info('data loading successful')
        logging.info('data transformation module starting') 
        data_transformer = DataTransformation()
        train_arr,test_arr = data_transformer.Transformer(train_path,test_path) 
        logging.info('data transformation successful')  
        print('data transformation successful') 

    except Exception as e :
        logging.error(e)
        raise CustomException(e,sys)  
    


if __name__ == "__main__":

    train_pip = Trainig_pipeline() 