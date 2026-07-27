import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:

    def run_pipeline(self):
        try:
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_path, test_path
            )

            trainer = ModelTrainer()
            trainer.initiate_model_trainer(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()