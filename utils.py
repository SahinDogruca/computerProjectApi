"""
Utility functions for label reading and chart creation
"""

from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import cv2
import supervision as sv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def read_yolo_labels(
    label_path: Path, image_shape: Tuple
) -> Tuple[sv.Detections, bool]:

    height, width = image_shape[:2]

    if not label_path.exists():
        return sv.Detections.empty(), False

    xyxy_list = []
    class_ids = []
    polygons = []
    is_segmentation = False

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            # Segmentation format (more than 5 values)
            if len(parts) > 5:
                is_segmentation = True

                coords = [float(x) for x in parts[1:]]

                # Convert normalized coords to pixels
                polygon_points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * width)
                    y = int(coords[i + 1] * height)
                    polygon_points.append([x, y])

                polygon_points = np.array(polygon_points, dtype=np.int32)
                polygons.append(polygon_points)

                # Calculate bounding box from polygon
                x_min = polygon_points[:, 0].min()
                y_min = polygon_points[:, 1].min()
                x_max = polygon_points[:, 0].max()
                y_max = polygon_points[:, 1].max()

            else:
                # Bbox format (YOLO: x_center, y_center, width, height)
                x_center, y_center, w, h = map(float, parts[1:5])

                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)

            xyxy_list.append([x_min, y_min, x_max, y_max])
            class_ids.append(class_id)

    if len(xyxy_list) == 0:
        return sv.Detections.empty(), is_segmentation

    detections = sv.Detections(
        xyxy=np.array(xyxy_list),
        class_id=np.array(class_ids),
    )

    # Add masks if segmentation
    if is_segmentation and len(polygons) > 0:
        masks = []
        for polygon in polygons:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            masks.append(mask.astype(bool))
        detections.mask = np.array(masks)

    return detections, is_segmentation


def create_confidence_chart(
    detections: sv.Detections,
    class_names: Dict[int, str],
    conf_threshold: float = 0.5,
    chart_conf_threshold: float = 0.001,
    top_k: int = 10,
) -> str:

    # Filter by chart threshold
    mask = detections.confidence >= chart_conf_threshold
    filtered_detections = detections[mask]

    if len(filtered_detections) == 0:
        return None

    predictions_list = []
    for class_id, conf in zip(
        filtered_detections.class_id, filtered_detections.confidence
    ):
        class_name = class_names.get(int(class_id), f"Class {class_id}")
        predictions_list.append((class_name, float(conf)))

    # Sort by confidence (high to low)
    predictions_list.sort(key=lambda x: x[1], reverse=True)

    # Take top K
    predictions_list = predictions_list[:top_k]

    if len(predictions_list) == 0:
        return None

    classes = [item[0] for item in predictions_list]
    confidences = [item[1] for item in predictions_list]

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = [
        "#7B3F9E" if conf >= conf_threshold else "#3498db" for conf in confidences
    ]

    bars = ax.bar(
        range(len(classes)),
        confidences,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{conf:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Threshold line
    ax.axhline(
        y=conf_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {conf_threshold}",
        zorder=10,
    )

    ax.set_xlabel("Tahmin (En Yüksek Confidence)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Confidence Değeri", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {len(predictions_list)} Tahmin Confidence Değerleri",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=11)

    # Y-axis format
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)

    return chart_base64
