import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tensorflow as tf

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    image_height: int = 200
    image_width: int = 200
    batch_size: int = 16
    validation_split: float = 0.2
    seed: int = 42
    normalization_scale: float = 280.0
    artifact_dir: str = os.path.join("artifacts", "data_transformation")
    class_names_path: str = os.path.join(artifact_dir, "class_names.json")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self._dataset_options = self._build_dataset_options()

    def _build_dataset_options(self) -> tf.data.Options:
        options = tf.data.Options()
        cpu_count = max(1, os.cpu_count() or 1)
        options.experimental_threading.private_threadpool_size = max(1, cpu_count // 2)
        options.experimental_threading.max_intra_op_parallelism = max(1, cpu_count - 2)
        options.experimental_deterministic = False
        return options

    def _prepare_datasets(self, dataset_paths: Dict[str, str]):
        try:
            train_dir = dataset_paths["train_dir"]
            test_dir = dataset_paths["test_dir"]
        except KeyError as exc:
            raise CustomException(f"Missing dataset directory key: {exc}", sys)

        logging.info("Creating TensorFlow datasets from image directories")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=self.config.validation_split,
            subset="training",
            seed=self.config.seed,
            image_size=(self.config.image_height, self.config.image_width),
            batch_size=self.config.batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=self.config.validation_split,
            subset="validation",
            seed=self.config.seed,
            image_size=(self.config.image_height, self.config.image_width),
            batch_size=self.config.batch_size
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            shuffle=False,
            image_size=(self.config.image_height, self.config.image_width),
            batch_size=self.config.batch_size
        )

        return train_ds, val_ds, test_ds

    def initiate_data_transformation(self, dataset_paths: Dict[str, str]):
        try:
            train_ds, val_ds, test_ds = self._prepare_datasets(dataset_paths)

            class_names: List[str] = train_ds.class_names
            os.makedirs(self.config.artifact_dir, exist_ok=True)

            with open(self.config.class_names_path, "w", encoding="utf-8") as fp:
                json.dump(class_names, fp, indent=2)

            logging.info("Class names saved for downstream prediction pipeline")

            autotune = tf.data.AUTOTUNE
            augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.02),
                    tf.keras.layers.RandomZoom(0.1),
                ],
                name="data_augmentation"
            )
            normalization_layer = tf.keras.layers.Rescaling(1.0 / self.config.normalization_scale)

            def preprocess(dataset, training=False):
                ds = dataset
                if training:
                    ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                                num_parallel_calls=autotune)
                ds = ds.map(lambda x, y: (normalization_layer(x), y),
                            num_parallel_calls=autotune)
                ds = ds.cache()
                ds = ds.prefetch(buffer_size=autotune)
                return ds.with_options(self._dataset_options)

            train_ds = preprocess(train_ds, training=True)
            val_ds = preprocess(val_ds)
            test_ds = preprocess(test_ds)

            return train_ds, val_ds, test_ds, class_names

        except Exception as exc:
            raise CustomException(exc, sys)
