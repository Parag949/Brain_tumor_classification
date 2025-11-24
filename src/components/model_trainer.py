import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import tensorflow as tf

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.keras")
    history_path: str = os.path.join("artifacts", "training_history.json")
    metrics_path: str = os.path.join("artifacts", "evaluation_metrics.json")
    epochs: int = 10
    learning_rate: float = 1e-3
    intra_op_threads: Optional[int] = None
    inter_op_threads: Optional[int] = None
    device: str = "/CPU:0"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self._configure_tensorflow_threads()

    def _configure_tensorflow_threads(self):
        cpu_count = max(1, os.cpu_count() or 1)
        intra = self.config.intra_op_threads or max(1, cpu_count - 2)
        inter = self.config.inter_op_threads or max(1, cpu_count // 2)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
            logging.info(
                "Configured TensorFlow threading: intra_op=%s inter_op=%s (cpu_count=%s)",
                intra,
                inter,
                cpu_count,
            )
        except RuntimeError:
            logging.info(
                "TensorFlow runtime already initialised; keeping existing threading config "
                "(current intra_op=%s inter_op=%s)",
                tf.config.threading.get_intra_op_parallelism_threads(),
                tf.config.threading.get_inter_op_parallelism_threads(),
            )

    def _build_model(self, input_shape, num_classes: int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model

    def initiate_model_trainer(self, train_ds, val_ds, test_ds, class_names: List[str]) -> Dict[str, float]:
        try:
            for dataset in (train_ds, val_ds, test_ds):
                if dataset is None:
                    raise CustomException("Received an empty dataset during training", sys)

            input_shape = (train_ds.element_spec[0].shape[1],
                           train_ds.element_spec[0].shape[2],
                           train_ds.element_spec[0].shape[3])
            num_classes = len(class_names)

            model = self._build_model(input_shape, num_classes)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-5,
                    verbose=1
                )
            ]

            logging.info("Starting model training")
            with tf.device(self.config.device):
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.config.epochs,
                    callbacks=callbacks
                )

            logging.info("Evaluating model on test dataset")
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            model.save(self.config.trained_model_file_path)

            history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
            metrics_dict = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy)
            }

            with open(self.config.history_path, "w", encoding="utf-8") as fp:
                json.dump(history_dict, fp, indent=2)

            with open(self.config.metrics_path, "w", encoding="utf-8") as fp:
                json.dump(metrics_dict, fp, indent=2)

            logging.info("Model training completed and artifacts stored")

            return metrics_dict

        except Exception as exc:
            raise CustomException(exc, sys)
