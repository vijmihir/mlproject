import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder  

sys.path.append('/Users/mihir.vij/Desktop/mlproject')
from src.exception import CustomException

from src.logger import logging
from src.utils import save_object

@dataclass # Designed to hold only data values. It is called a decorator. When we do this we do not need to write __init__ functions 
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact',"preprocessor.pkl")
    

class DataTransformation:
    def __init__ (self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
         This function is resposible for data transformation
         - In the try block we are doing the following:
            Defining the numerical features and categorical features 
            Creating a numerical pipeline with imputer and scaler 
            Creating a categorical pipeline with imputer, one-hot encoder and scaler 
                Using Pipeline from sklearn.pipeline
            Creating preprocessor using ColumnTransformer where we give our numerical and categorical pipeline followed by numerical 
                and categorical features.
            Returning the preprocessor

        '''
        try:
            numerical_features = ["writing score","reading score"]
            categorical_features = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median'))
                    , ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent'))
                    , ("one-hot encoder", OneHotEncoder())
                    , ("scaler", StandardScaler())
                ]
            )
            logging.info('Categorical and numerical columns standard scaling completed')
            preprocessor = ColumnTransformer(
                [
                    ("numerical pipeline", numerical_pipeline, numerical_features)
                    ("categorical pipeline", categorical_pipeline, categorical_features)
                ]
            )
        
            return preprocessor 
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test completed')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math score"

            numerical_features = ["writing score","reading score"]
            categorical_features = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
            
            input_feature_train_df = train_df.drop(columns = target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('save preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj

            )
            return {
                train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path
            }
        except Exception as e:
            raise CustomException(e,sys)