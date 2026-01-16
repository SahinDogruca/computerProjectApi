"""
Model management module - handles loading, prediction, and model operations
"""

from pathlib import Path
from typing import Tuple, Any, List, Dict
import time
import numpy as np
import cv2
import supervision as sv
from os import chdir


class ModelManager:
    """Manages model loading, prediction, and related operations"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.base_dir = models_dir.parent

    def load_model(self, model_name: str) -> Tuple[Any, bool]:
        """
        Load a model by name and return the model instance and segmentation flag

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (model, is_segmentation)
        """
        is_segmentation = False

        if model_name.startswith("yolo"):
            # Handle YOLO v12 special case
            if model_name.startswith("yolov12"):
                chdir("yolov12")

            from ultralytics import YOLO

            model = YOLO(self.models_dir / f"{model_name}.pt")
            is_segmentation = "seg" in model_name.lower()

        elif model_name.startswith("rt-detr"):
            from ultralytics import RTDETR

            model = RTDETR(self.models_dir / f"{model_name}.pt")
            is_segmentation = False

        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        return model, is_segmentation

    def predict(
        self, model: Any, source: Any, conf_threshold: float = 0.001, imgsz: int = 1536
    ) -> Tuple[Any, float]:
        """
        Run prediction on source

        Args:
            model: Loaded model instance
            source: Image path or numpy array
            conf_threshold: Confidence threshold for chart
            imgsz: Image size for inference

        Returns:
            Tuple of (results, inference_time)
        """
        start_time = time.time()
        results = model.predict(
            source=source, imgsz=imgsz, device="cpu", conf=conf_threshold
        )
        inference_time = time.time() - start_time

        # Change back to base directory
        chdir(self.base_dir)

        return results, inference_time

    @staticmethod
    def get_detections(result: Any) -> sv.Detections:
        """
        Convert Ultralytics result to Supervision Detections

        Args:
            result: Ultralytics prediction result

        Returns:
            Supervision Detections object
        """
        return sv.Detections.from_ultralytics(result)

    @staticmethod
    def format_predictions(
        detections: sv.Detections, class_names: Dict[int, str], is_segmentation: bool
    ) -> List[Dict]:
        """
        Format detections into a list of prediction dictionaries

        Args:
            detections: Supervision Detections object
            class_names: Dictionary mapping class IDs to names
            is_segmentation: Whether the model is segmentation

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for i in range(len(detections)):
            prediction = {
                "class": class_names[int(detections.class_id[i])],
                "confidence": float(detections.confidence[i]),
                "bbox": detections.xyxy[i].tolist(),
                "has_mask": (detections.mask is not None if is_segmentation else False),
            }
            predictions.append(prediction)

        return predictions

    @staticmethod
    async def read_uploaded_image(file) -> np.ndarray:
        """
        Read an uploaded image file

        Args:
            file: UploadFile object

        Returns:
            Image as numpy array
        """
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image file")

        return img

    def list_models(self) -> List[str]:
        """
        List all available models

        Returns:
            List of model names (without extension)
        """
        models = []
        for model_file in self.models_dir.iterdir():
            if (
                model_file.is_file()
                and model_file.suffix in [".pt", ".pth"]
                and not model_file.stem.endswith("config")
            ):
                models.append(model_file.stem)

        return models
