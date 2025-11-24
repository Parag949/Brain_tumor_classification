import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from src.exception import CustomException


@dataclass
class PredictionResult:
    label: str
    confidence: float


@dataclass
class BrainScan:
    """Carries a preprocessed MRI slice ready for inference."""

    image: np.ndarray
    image_size: Tuple[int, int] = (200, 200)

    @classmethod
    def from_file_storage(cls, storage, image_size: Tuple[int, int] = (200, 200)):
        """Create a scan object from a Flask FileStorage."""
        file_bytes = storage.read()
        if not file_bytes:
            raise CustomException("Uploaded file is empty", sys)
        return cls.from_bytes(file_bytes, image_size=image_size)

    @classmethod
    def from_bytes(cls, data: bytes, image_size: Tuple[int, int] = (200, 200)):
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                img = img.resize(image_size)
                array = np.asarray(img, dtype=np.float32)
        except Exception as exc:
            raise CustomException(f"Unable to process the uploaded image: {exc}", sys)

        return cls(image=array, image_size=image_size)

    def to_model_input(self, normalization_scale: float) -> np.ndarray:
        normalized = self.image / normalization_scale
        return np.expand_dims(normalized, axis=0)


class PredictPipeline:
    def __init__(self,
                 model_path: str = os.path.join("artifacts", "model.keras"),
                 class_names_path: str = os.path.join(
                     "artifacts", "data_transformation", "class_names.json"
                 ),
                 normalization_scale: float = 280.0,
                 image_size: Tuple[int, int] = (200, 200)):

        self.model_path = model_path
        self.class_names_path = class_names_path
        self.normalization_scale = normalization_scale
        self.image_size = image_size

        self._model = None
        self._class_names = None

    def _load_artifacts(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise CustomException(
                    f"Model artifact not found at {self.model_path}. Train the pipeline first.",
                    sys
                )
            self._model = tf.keras.models.load_model(self.model_path)

        if self._class_names is None:
            if not os.path.exists(self.class_names_path):
                raise CustomException(
                    f"Class definition file missing at {self.class_names_path}", sys
                )
            with open(self.class_names_path, "r", encoding="utf-8") as fp:
                self._class_names = json.load(fp)

    def predict(self, scan: BrainScan) -> PredictionResult:
        try:
            self._load_artifacts()
            model_input = scan.to_model_input(self.normalization_scale)

            logits = self._model.predict(model_input, verbose=0)[0]
            probabilities = tf.nn.softmax(logits).numpy()
            top_index = int(np.argmax(probabilities))

            label = self._class_names[top_index] if self._class_names else str(top_index)
            confidence = float(probabilities[top_index]) if probabilities.size else 0.0

            return PredictionResult(label=label, confidence=confidence)
        except Exception as exc:
            raise CustomException(exc, sys)
