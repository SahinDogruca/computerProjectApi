from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import time
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

MODELS = [
    "yolov8x-seg",
    "yolov8x-seg-pretrained",
    "yolov12l-seg",
    "yolov12l-seg-pretrained",
    "rt-detrv4",
    "rt-detrv4-pretrained",
]


app = FastAPI(title="predictApi", version="1.0.0")

origins = ["http://localhost:5173"]  # vite frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE
    allow_headers=["*"],  # Authorization, Content-Type vs
)


@app.get("/")
def root():
    return {"status": "app is running"}


def load_gt_masks(label_path, img_shape):
    h, w = img_shape[:2]
    gt_objects = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

            poly = np.zeros_like(coords)
            poly[:, 0] = coords[:, 0] * w
            poly[:, 1] = coords[:, 1] * h
            poly = poly.astype(np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)

            gt_objects.append({"class": cls, "poly": poly, "mask": mask})

    return gt_objects


def load_pred_masks(result, conf_threshold=0.5):
    h, w = result.orig_img.shape[:2]
    pred_objects = []

    if result.masks is None:
        return []

    for i in range(len(result.boxes)):
        conf = float(result.boxes.conf[i])

        # Confidence threshold filtresi
        if conf < conf_threshold:
            continue

        cls = int(result.boxes.cls[i])
        poly = result.masks.xy[i].astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)

        pred_objects.append({"class": cls, "conf": conf, "poly": poly, "mask": mask})

    return pred_objects


def visualize_gt_pred_overlay(label_path, result):
    img = result.orig_img.copy()
    h, w = img.shape[:2]

    gt_objs = load_gt_masks(label_path, img.shape)
    pred_objs = load_pred_masks(result)

    # -------------------------
    # GT IMAGE
    # -------------------------
    gt_img = img.copy()
    for g in gt_objs:
        cv2.fillPoly(gt_img, [g["poly"]], (0, 255, 0))
        x, y = g["poly"][0]
        cv2.putText(
            gt_img,
            result.names[g["class"]],
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # -------------------------
    # PRED IMAGE
    # -------------------------
    pred_img = img.copy()
    for p in pred_objs:
        cv2.fillPoly(pred_img, [p["poly"]], (0, 0, 255))
        x, y = p["poly"][0]
        label = f"{result.names[p['class']]} {p['conf']:.2f}"
        cv2.putText(
            pred_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # -------------------------
    # OVERLAY
    # -------------------------
    overlay = img.copy()

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for g in gt_objs:
        gt_mask |= g["mask"]
    for p in pred_objs:
        pred_mask |= p["mask"]

    tp_mask = gt_mask & pred_mask

    # GT only → green
    overlay[(gt_mask == 1) & (tp_mask == 0)] = [0, 255, 0]
    # Pred only → red
    overlay[(pred_mask == 1) & (tp_mask == 0)] = [0, 0, 255]
    # True Positive → blue
    overlay[tp_mask == 1] = [255, 0, 0]

    return gt_img, pred_img, overlay


import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import io
import base64


def plot_confidence_chart(result, conf_threshold=0.001, predict_threshold=0.5):
    """
    Tahminlerin confidence değerlerini bar chart olarak döndürür.

    Args:
        result: YOLO prediction result
        conf_threshold: Gösterilecek minimum confidence değeri
        return_base64: True ise base64 string, False ise bytes döner

    Returns:
        base64 string veya bytes (format: PNG)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    # Confidence ve class bilgilerini topla
    confidences = []
    class_names = []

    for i in range(len(result.boxes)):
        conf = float(result.boxes.conf[i])

        if conf >= conf_threshold:
            cls = int(result.boxes.cls[i])
            class_name = result.names[cls]

            confidences.append(conf)
            class_names.append(f"{class_name}\n({conf:.3f})")

    if len(confidences) == 0:
        return None

    # Bar chart oluştur
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(confidences)), confidences, color="steelblue", alpha=0.8)

    # Her bar için farklı renk
    colors = plt.cm.viridis(np.linspace(0, 1, len(confidences)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel("Sınıf Adı", fontsize=12, fontweight="bold")
    ax.set_ylabel("Confidence Değeri", fontsize=12, fontweight="bold")
    ax.set_title("Tahmin Confidence Değerleri", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(confidences)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Threshold çizgisi ekle
    ax.axhline(
        y=predict_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {predict_threshold}",
    )
    ax.legend()

    plt.tight_layout()

    # Memory'de PNG olarak kaydet
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


@app.get("/predict/test")
def predict_test(model_name: str, image_name: str):

    if not model_name in MODELS:
        return {"error": "model not found in models"}

    if model_name.startswith("yolo"):
        from ultralytics import YOLO

        model = YOLO(MODELS_DIR / f"{model_name}.pt")

        image_path = TEST_DIR / "images" / image_name
        image_label_path = TEST_DIR / "labels" / f"{image_name.split('.')[0]}.txt"

        start_time = time.time()
        results = model.predict(source=image_path, imgsz=1536, device="cpu", conf=0.001)
        inference_time = time.time() - start_time

        result = results[0]
        print(result)

        img_gt, img_pred, img_overlay = visualize_gt_pred_overlay(
            image_label_path, result
        )

        _, gt_buffer = cv2.imencode(".png", img_gt)
        _, pred_buffer = cv2.imencode(".png", img_pred)
        _, overlay_buffer = cv2.imencode(".png", img_overlay)

        confidence_chart = plot_confidence_chart(result)

        return JSONResponse(
            {
                # "gt_image": base64.b64encode(gt_buffer).decode("utf-8"),
                # "pred_image": base64.b64encode(pred_buffer).decode("utf-8"),
                # "overlay_image": base64.b64encode(overlay_buffer).decode("utf-8"),
                # "confidence_chart": confidence_chart,
                "inference_time": inference_time,
                "predictions": [
                    {
                        "class": result.names[int(result.boxes.cls[i])],
                        "confidence": float(result.boxes.conf[i]),
                    }
                    for i in range(len(result.boxes))
                ],
            }
        )
