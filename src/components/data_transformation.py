import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.imputer import SimpleImputer
from sklearn.imputer import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logging import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['mileage','make_year','price','no_of_owners','registration_year']

            categorical_columns = ['city','make','model','variant','fuel_type','color','body_type','transmission','latest_publish_date']

            num_pipeline = Pipelineipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scalar",StandardScaler())])

            cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("one_hot_encoder", OneHotEncoder()),("scaler",StandardScalar(with_mean=False))])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer([("num_pipeline",num_pipeline,numerical_columns),("cat_pipeline" , cat_pipeline , categorical_columns)])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_column_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_column_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_arr)

            train_arr = np.c_[input_feature_train_arr, np.array(target_column_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_column_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)


