import json
import os
import sys
from typing import Any, Dict

import dill

from src.exception import CustomException


def save_object(file_path: str, obj: Any) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as exc:
        raise CustomException(exc, sys)


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as exc:
        raise CustomException(exc, sys)


def save_json(file_path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
    except Exception as exc:
        raise CustomException(exc, sys)


def load_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        raise CustomException(exc, sys)