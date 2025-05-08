import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = data_transformation_config()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading score' , 'writing score']
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("Imputer" , SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer" ,SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder" , OneHotEncoder())
                ]
            )

            logging.info("Numerical features standard scaling comleted")
            logging.info("Categorical features one hot encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline" , numerical_pipeline , numerical_features),
                    ("cat_pipeline" , cat_pipeline , categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self , train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed ")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name= "math score"

            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name] , axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e :
            raise CustomException(e,sys)
        