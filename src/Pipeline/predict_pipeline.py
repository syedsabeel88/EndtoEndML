import sys,os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.Components.data_transformation import DataTransformation, ColumnTransformer


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading Pickle file")
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            print("After Loading")
            print("preprocessor",preprocessor)
            data_scaled = preprocessor.transform(features)
            print(data_scaled)
            preds = model.predict(data_scaled)
            print(preds)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



#below class is to map data from frontend html webapplication to backend
class CustomData:
    def __init__(self, gender,race_ethnicity:str,parental_level_of_education,
                 lunch,test_preparation_course,reading_score:int,writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch =lunch
        self.test_preparation_course =test_preparation_course
        self.reading_score =reading_score
        self.writing_score = writing_score

    #below function will return data in the form of dataframe
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        