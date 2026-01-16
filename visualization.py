"""
Visualization module - handles all image annotation and visualization
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
import supervision as sv
import base64
from io import BytesIO
from utils import read_yolo_labels


class Visualizer:

    def __init__(self):
        self.mask_colors = sv.ColorPalette.from_hex(
            ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        )
        self.gt_color = sv.Color.from_hex("#00FF00")  # Green
        self.pred_color = sv.Color.from_hex("#FF0000")  # Red

    def _create_annotators(self, is_segmentation: bool):
        annotators = {
            "box": sv.BoxAnnotator(thickness=2),
            "label": sv.LabelAnnotator(text_scale=0.5, text_thickness=1),
        }

        if is_segmentation:
            annotators["mask"] = sv.MaskAnnotator(color=self.mask_colors)

        return annotators

    def _add_title(self, img: np.ndarray, title: str, color: Tuple[int, int, int]):
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

    def _encode_image(self, img: np.ndarray) -> str:
        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode("utf-8")

    def _annotate_detections(
        self,
        img: np.ndarray,
        detections: sv.Detections,
        labels: list,
        annotators: dict,
        use_mask: bool = False,
    ) -> np.ndarray:
        if len(detections) == 0:
            return img

        # Apply mask if needed
        if use_mask and "mask" in annotators and detections.mask is not None:
            img = annotators["mask"].annotate(img, detections)

        img = annotators["box"].annotate(img, detections)
        img = annotators["label"].annotate(img, detections, labels)

        return img

    def create_gt_pred_overlay(
        self,
        image_path: Path,
        label_path: Path,
        predictions: sv.Detections,
        class_names: Dict[int, str],
        conf_threshold: float = 0.5,
        is_segmentation: bool = False,
    ) -> Dict[str, str]:

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Read ground truth
        gt_detections, gt_is_seg = read_yolo_labels(label_path, image.shape)

        # Filter predictions by threshold
        filtered_predictions = predictions[predictions.confidence >= conf_threshold]

        # Create annotators
        annotators = self._create_annotators(is_segmentation or gt_is_seg)

        # === GROUND TRUTH IMAGE ===
        img_gt = image.copy()
        if len(gt_detections) > 0:
            gt_labels = [
                class_names.get(int(class_id), f"Class {class_id}")
                for class_id in gt_detections.class_id
            ]
            img_gt = self._annotate_detections(
                img_gt, gt_detections, gt_labels, annotators, use_mask=gt_is_seg
            )
        img_gt = self._add_title(img_gt, "Ground Truth", (0, 255, 0))

        # === PREDICTION IMAGE ===
        img_pred = image.copy()
        if len(filtered_predictions) > 0:
            pred_labels = [
                f"{class_names.get(int(filtered_predictions.class_id[i]), f'Class {filtered_predictions.class_id[i]}')} "
                f"{filtered_predictions.confidence[i]:.2f}"
                for i in range(len(filtered_predictions))
            ]
            img_pred = self._annotate_detections(
                img_pred,
                filtered_predictions,
                pred_labels,
                annotators,
                use_mask=is_segmentation,
            )
        img_pred = self._add_title(
            img_pred, f"Predictions (conf >= {conf_threshold})", (255, 0, 0)
        )

        # === OVERLAY IMAGE ===
        img_overlay = self._create_overlay(
            image,
            gt_detections,
            filtered_predictions,
            class_names,
            gt_is_seg,
            is_segmentation,
        )

        return {
            "gt_image": self._encode_image(img_gt),
            "pred_image": self._encode_image(img_pred),
            "overlay_image": self._encode_image(img_overlay),
        }

    def _create_overlay(
        self,
        image: np.ndarray,
        gt_detections: sv.Detections,
        pred_detections: sv.Detections,
        class_names: Dict[int, str],
        gt_is_seg: bool,
        pred_is_seg: bool,
    ) -> np.ndarray:
        """Create overlay image with GT (green) and Predictions (red)"""
        img_overlay = image.copy()

        # GT annotators (green)
        gt_box_annotator = sv.BoxAnnotator(thickness=2, color=self.gt_color)
        gt_label_annotator = sv.LabelAnnotator(
            text_scale=0.4, text_thickness=1, text_color=sv.Color.from_hex("#FFFFFF")
        )

        # Annotate GT
        if len(gt_detections) > 0:
            gt_labels = [
                f"GT: {class_names.get(int(cid), f'C{cid}')}"
                for cid in gt_detections.class_id
            ]

            if gt_is_seg and gt_detections.mask is not None:
                gt_mask_annotator = sv.MaskAnnotator(
                    color=sv.ColorPalette.from_hex(["#00FF00"]), opacity=0.3
                )
                img_overlay = gt_mask_annotator.annotate(img_overlay, gt_detections)

            img_overlay = gt_box_annotator.annotate(img_overlay, gt_detections)
            img_overlay = gt_label_annotator.annotate(
                img_overlay, gt_detections, gt_labels
            )

        # Prediction annotators (red)
        pred_box_annotator = sv.BoxAnnotator(thickness=2, color=self.pred_color)
        pred_label_annotator = sv.LabelAnnotator(
            text_scale=0.4, text_thickness=1, text_color=sv.Color.from_hex("#FFFFFF")
        )

        # Annotate predictions
        if len(pred_detections) > 0:
            pred_labels = [
                f"Pred: {class_names.get(int(pred_detections.class_id[i]), f'C{pred_detections.class_id[i]}')} "
                f"{pred_detections.confidence[i]:.2f}"
                for i in range(len(pred_detections))
            ]

            if pred_is_seg and pred_detections.mask is not None:
                pred_mask_annotator = sv.MaskAnnotator(
                    color=sv.ColorPalette.from_hex(["#FF0000"]), opacity=0.3
                )
                img_overlay = pred_mask_annotator.annotate(img_overlay, pred_detections)

            img_overlay = pred_box_annotator.annotate(img_overlay, pred_detections)
            img_overlay = pred_label_annotator.annotate(
                img_overlay, pred_detections, pred_labels
            )

        # Add title
        img_overlay = self._add_title(
            img_overlay, "Overlay (Green=GT, Red=Pred)", (255, 255, 255)
        )

        return img_overlay

    def create_gt_pred_overlay_from_detections(
        self,
        img: np.ndarray,
        gt_detections: sv.Detections,
        pred_detections: sv.Detections,
        class_names: Dict[int, str],
        conf_threshold: float = 0.5,
        is_segmentation: bool = False,
        gt_is_seg: bool = False,
    ) -> Dict[str, str]:
        """Create visualizations when GT is provided as detections"""

        # Filter predictions
        filtered_predictions = pred_detections[
            pred_detections.confidence >= conf_threshold
        ]

        # Create annotators
        annotators = self._create_annotators(is_segmentation or gt_is_seg)

        # === GROUND TRUTH IMAGE ===
        img_gt = img.copy()
        if len(gt_detections) > 0:
            gt_labels = [
                class_names.get(int(class_id), f"Class {class_id}")
                for class_id in gt_detections.class_id
            ]
            img_gt = self._annotate_detections(
                img_gt, gt_detections, gt_labels, annotators, use_mask=gt_is_seg
            )
        img_gt = self._add_title(img_gt, "Ground Truth", (0, 255, 0))

        # === PREDICTION IMAGE ===
        img_pred = img.copy()
        if len(filtered_predictions) > 0:
            pred_labels = [
                f"{class_names.get(int(filtered_predictions.class_id[i]), f'Class {filtered_predictions.class_id[i]}')} "
                f"{filtered_predictions.confidence[i]:.2f}"
                for i in range(len(filtered_predictions))
            ]
            img_pred = self._annotate_detections(
                img_pred,
                filtered_predictions,
                pred_labels,
                annotators,
                use_mask=is_segmentation,
            )
        img_pred = self._add_title(
            img_pred, f"Predictions (conf >= {conf_threshold})", (255, 0, 0)
        )

        # === OVERLAY IMAGE ===
        img_overlay = self._create_overlay(
            img,
            gt_detections,
            filtered_predictions,
            class_names,
            gt_is_seg,
            is_segmentation,
        )

        return {
            "gt_image": self._encode_image(img_gt),
            "pred_image": self._encode_image(img_pred),
            "overlay_image": self._encode_image(img_overlay),
        }

    def create_pred_only(
        self,
        img: np.ndarray,
        predictions: sv.Detections,
        class_names: Dict[int, str],
        conf_threshold: float = 0.5,
        is_segmentation: bool = False,
    ) -> Dict[str, str]:
        """Create prediction-only visualization (no GT)"""

        # Filter predictions
        filtered_predictions = predictions[predictions.confidence >= conf_threshold]

        # Create annotators
        annotators = self._create_annotators(is_segmentation)

        # Create prediction image
        img_pred = img.copy()
        if len(filtered_predictions) > 0:
            pred_labels = [
                f"{class_names.get(int(filtered_predictions.class_id[i]), f'Class {filtered_predictions.class_id[i]}')} "
                f"{filtered_predictions.confidence[i]:.2f}"
                for i in range(len(filtered_predictions))
            ]
            img_pred = self._annotate_detections(
                img_pred,
                filtered_predictions,
                pred_labels,
                annotators,
                use_mask=is_segmentation,
            )
        img_pred = self._add_title(
            img_pred, f"Predictions (conf >= {conf_threshold})", (255, 0, 0)
        )

        return {
            "pred_image": self._encode_image(img_pred),
        }

    async def process_gt_file(self, gt_file, image_shape: Tuple[int, int, int]) -> Dict:
        try:
            gt_content = await gt_file.read()
            gt_text = gt_content.decode("utf-8")

            # Write to temp file
            temp_gt_path = Path("/tmp/temp_gt.txt")
            with open(temp_gt_path, "w") as f:
                f.write(gt_text)

            gt_detections, gt_is_seg = read_yolo_labels(temp_gt_path, image_shape)

            temp_gt_path.unlink()

            return {
                "detections": gt_detections,
                "is_segmentation": gt_is_seg,
                "error": None,
            }

        except Exception as e:
            return {
                "detections": None,
                "is_segmentation": False,
                "error": f"GT file parse error: {str(e)}",
            }
