import json
import os
import sys
from dataclasses import dataclass
from typing import Dict

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    dataset_source_dir: str = os.getenv(
        "BRAIN_TUMOR_DATA_DIR",
        os.path.join("brain", "Brain-Tumor-Classification-DataSet-master")
    )
    training_folder_name: str = "Training"
    testing_folder_name: str = "Testing"
    artifact_dir: str = os.path.join("artifacts", "data_ingestion")
    metadata_path: str = os.path.join(artifact_dir, "dataset_paths.json")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def _resolve_dataset_root(self) -> str:
        candidates = [self.config.dataset_source_dir]
        cwd_candidate = os.path.join("Brain-Tumor-Classification-DataSet-master")
        if cwd_candidate not in candidates:
            candidates.append(cwd_candidate)

        for path in candidates:
            if path and os.path.isdir(path):
                return path

        raise CustomException(
            "Unable to locate the Brain Tumor dataset. Set BRAIN_TUMOR_DATA_DIR or place the "
            "files under 'Brain-Tumor-Classification-DataSet-master/'.",
            sys
        )

    def initiate_data_ingestion(self) -> Dict[str, str]:
        logging.info("Entered the data ingestion method for brain tumor dataset")
        try:
            dataset_root = self._resolve_dataset_root()
            train_dir = os.path.join(dataset_root, self.config.training_folder_name)
            test_dir = os.path.join(dataset_root, self.config.testing_folder_name)

            if not os.path.isdir(train_dir):
                raise CustomException(
                    f"Training directory not found at {train_dir}. Please mount the dataset.", sys
                )

            if not os.path.isdir(test_dir):
                raise CustomException(
                    f"Testing directory not found at {test_dir}. Please mount the dataset.", sys
                )

            os.makedirs(self.config.artifact_dir, exist_ok=True)

            dataset_paths = {
                "train_dir": os.path.abspath(train_dir),
                "test_dir": os.path.abspath(test_dir)
            }

            with open(self.config.metadata_path, "w", encoding="utf-8") as fp:
                json.dump(dataset_paths, fp, indent=2)

            logging.info("Dataset directories located and metadata saved")

            return dataset_paths

        except Exception as exc:
            raise CustomException(exc, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    ingestion = DataIngestion()
    paths = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_ds, val_ds, test_ds, class_names = transformer.initiate_data_transformation(paths)

    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer(train_ds, val_ds, test_ds, class_names)
    print(metrics)
