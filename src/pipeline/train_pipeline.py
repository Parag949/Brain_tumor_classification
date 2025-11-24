from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def run_training():
	ingestion = DataIngestion()
	dataset_paths = ingestion.initiate_data_ingestion()

	transformer = DataTransformation()
	train_ds, val_ds, test_ds, class_names = transformer.initiate_data_transformation(dataset_paths)

	trainer = ModelTrainer()
	metrics = trainer.initiate_model_trainer(train_ds, val_ds, test_ds, class_names)

	logging.info(f"Training finished with metrics: {metrics}")
	return metrics


if __name__ == "__main__":
	run_training()
