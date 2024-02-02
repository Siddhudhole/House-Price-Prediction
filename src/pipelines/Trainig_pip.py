import os 
import sys 
from src.Logger import logging 
from src.Exception import CustomException 
from src.components.Data_ingestion import DataIngestionConfig 
from src.components.Data_ingestion import DataIngestion 


class Trainig_pipeline :
    try :
        data_ingestion = DataIngestion()
        train_path,test_path = data_ingestion.load_data() 
        print(train_path)
        print(test_path)

    except Exception as e :
        logging.error(e)
        raise CustomException(e,sys)  
    


if __name__ == "__main__":

    train_pip = Trainig_pipeline() 