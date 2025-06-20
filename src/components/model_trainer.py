import os 
import sys 
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor ,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from  xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr):
        try:
            logging.info("Split training and test input data")
            X_train , X_test , y_train , y_test = (
                train_arr[: , :-1],
                test_arr[: , :-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "AdaBoost Regression": AdaBoostRegressor() ,
                "Random Forest Regression": RandomForestRegressor() ,
                "Decision Tree Regression": DecisionTreeRegressor() ,
                "GradientBoost Regression": GradientBoostingRegressor() ,
                "XGBoost Regression" : XGBRegressor() ,
                "CatBoost Regression": CatBoostRegressor(),
                "KNeighbours Regression": KNeighborsRegressor() 
            }

            params={
                "Decision Tree Regression": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regression":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoost Regression":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost Regression":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regression":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regression":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighbours Regression":{}
            }

            logging.info("Starting with model evaluation")

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            logging.info("Model report evaluated successfully")

            # To get model with best score 
            best_model_score = max(sorted(model_report.values()))

            # To get model name with best score 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e :
            raise CustomException(e,sys)

