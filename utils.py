import os 
import sys 
import pickle 
from src.Logger import logging 
from sklearn.metrics import r2_score 



def model_evalution(x_train,x_test,y_train,y_test,models)->dict:
    logging.info('model evaluation is starting')
    report = {}
    for i in range(len(models)):
        model = list(models.values())[i]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        score = r2_score(y_test,y_pred)
        report[list(models.keys())[i]] = score
    logging.info('model evaluation is completed') 
    return report 


def save_model(file_path:str,object):
    with open(file_path,'wb') as f :
        pickle.dump(object,f)
