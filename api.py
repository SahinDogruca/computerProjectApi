from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from os import chdir

import time
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
RTDETR_REPO = BASE_DIR / "RT-DETRv4"
RESULTS_DIR = BASE_DIR / "results"


# Hardcoded 5 class names (sizin dataset'inizin sınıfları)
CLASS_NAMES = [
    "dentigeroz kist",
    "keratokist",
    "radikuler kist",
    "ameloblastoma",
    "odontoma",
]

app = FastAPI(title="predictApi", version="1.0.0")

app.mount("/results", StaticFiles(directory="results"), name="results")

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "app is running"}


# ============================================================================
# UPLOAD HELPER FUNCTIONS
# ============================================================================


def parse_yolo_gt_from_string(gt_text, img_shape, class_names):
    """
    YOLO format string'i parse et (memory'den)

    Format: class_id x1 y1 x2 y2 ... (normalized polygon coordinates)
    """
    h, w = img_shape[:2]
    gt_objects = []

    for line in gt_text.strip().split("\n"):
        if not line.strip():
            continue

        parts = line.strip().split()
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

        poly = np.zeros_like(coords)
        poly[:, 0] = coords[:, 0] * w
        poly[:, 1] = coords[:, 1] * h
        poly = poly.astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)

        gt_objects.append(
            {
                "class": cls,
                "class_name": (
                    class_names[cls] if cls < len(class_names) else f"Class {cls}"
                ),
                "poly": poly,
                "mask": mask,
            }
        )

    return gt_objects


def parse_rtdetr_gt_from_json(gt_json, class_names):
    """
    Simplified COCO JSON'ı parse et

    Format:
    {
        "annotations": [
            {"class_id": 0, "bbox": [x1, y1, x2, y2]},
            {"class_id": 1, "bbox": [x1, y1, x2, y2]}
        ]
    }
    """
    gt_bboxes = []

    annotations = gt_json.get("annotations", [])

    for ann in annotations:
        class_id = ann["class_id"]
        bbox = ann["bbox"]  # [x1, y1, x2, y2]

        gt_bboxes.append(
            {
                "bbox": bbox,
                "class": class_id,
                "class_name": (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"Class {class_id}"
                ),
            }
        )

    return gt_bboxes


def visualize_gt_pred_overlay_from_objects(img, gt_objs, pred_objs, class_names_dict):
    """YOLO: GT ve Pred objects'ten overlay oluştur"""
    h, w = img.shape[:2]

    # GT IMAGE
    gt_img = img.copy()
    for g in gt_objs:
        cv2.fillPoly(gt_img, [g["poly"]], (0, 255, 0))
        x, y = g["poly"][0]
        cv2.putText(
            gt_img,
            g["class_name"],
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # PRED IMAGE
    pred_img = img.copy()
    for p in pred_objs:
        cv2.fillPoly(pred_img, [p["poly"]], (0, 0, 255))
        x, y = p["poly"][0]
        label = f"{class_names_dict[p['class']]} {p['conf']:.2f}"
        cv2.putText(
            pred_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # OVERLAY
    overlay = img.copy()
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for g in gt_objs:
        gt_mask |= g["mask"]
    for p in pred_objs:
        pred_mask |= p["mask"]

    tp_mask = gt_mask & pred_mask
    overlay[(gt_mask == 1) & (tp_mask == 0)] = [0, 255, 0]
    overlay[(pred_mask == 1) & (tp_mask == 0)] = [0, 0, 255]
    overlay[tp_mask == 1] = [255, 0, 0]

    return gt_img, pred_img, overlay


def visualize_pred_only(img, pred_objs, class_names_dict):
    """YOLO: Sadece prediction göster (GT yok)"""
    pred_img = img.copy()

    for p in pred_objs:
        cv2.fillPoly(pred_img, [p["poly"]], (0, 0, 255))
        x, y = p["poly"][0]
        label = f"{class_names_dict[p['class']]} {p['conf']:.2f}"
        cv2.putText(
            pred_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    return pred_img


def rtdetr_inference_image_from_array(model, img_array, cfg, device="cpu"):
    """
    RT-DETRv4 inference from numpy array (memory'den)

    Args:
        img_array: numpy array (BGR format from cv2)

    Returns:
        outputs, orig_img, orig_size
    """
    import torch

    orig_img = img_array.copy()
    orig_h, orig_w = orig_img.shape[:2]

    # Resize size config'ten al
    eval_size = cfg.yaml_cfg.get("eval_spatial_size", [960, 960])
    if isinstance(eval_size, int):
        eval_size = [eval_size, eval_size]
    resize_h, resize_w = eval_size

    # Preprocessing
    img = cv2.resize(orig_img, (resize_w, resize_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Original size tensor
    orig_size = torch.tensor([[orig_h, orig_w]], device=device)

    # Inference
    with torch.no_grad():
        outputs = model(img, orig_size)

    return outputs, orig_img, (orig_h, orig_w)


def visualize_rtdetr_pred_only(img, detections, class_names, conf_threshold=0.5):
    """RT-DETR: Sadece prediction göster (GT yok)"""
    pred_img = img.copy()

    for box, score, label_id in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        class_name = (
            class_names[label_id]
            if label_id < len(class_names)
            else f"Class {label_id}"
        )
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(
            pred_img,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return pred_img


def load_gt_masks(label_path, img_shape):
    """YOLO polygon format için ground truth masks yükle"""
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
    """YOLO prediction result'tan mask'leri çıkar"""
    h, w = result.orig_img.shape[:2]
    pred_objects = []

    if result.masks is None:
        return []

    for i in range(len(result.boxes)):
        conf = float(result.boxes.conf[i])

        if conf < conf_threshold:
            continue

        cls = int(result.boxes.cls[i])
        poly = result.masks.xy[i].astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)

        pred_objects.append({"class": cls, "conf": conf, "poly": poly, "mask": mask})

    return pred_objects


def visualize_gt_pred_overlay(label_path, result, conf_threshold=0.5):
    """YOLO segmentation için GT vs Prediction overlay"""
    img = result.orig_img.copy()
    h, w = img.shape[:2]

    gt_objs = load_gt_masks(label_path, img.shape)
    pred_objs = load_pred_masks(result, conf_threshold)

    # GT IMAGE
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

    # PRED IMAGE
    pred_img = img.copy()
    for p in pred_objs:
        cv2.fillPoly(pred_img, [p["poly"]], (0, 0, 255))
        x, y = p["poly"][0]
        label = f"{result.names[p['class']]} {p['conf']:.2f}"
        cv2.putText(
            pred_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # OVERLAY
    overlay = img.copy()
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for g in gt_objs:
        gt_mask |= g["mask"]
    for p in pred_objs:
        pred_mask |= p["mask"]

    tp_mask = gt_mask & pred_mask
    overlay[(gt_mask == 1) & (tp_mask == 0)] = [0, 255, 0]
    overlay[(pred_mask == 1) & (tp_mask == 0)] = [0, 0, 255]
    overlay[tp_mask == 1] = [255, 0, 0]

    return gt_img, pred_img, overlay


def plot_confidence_chart_yolo(result, conf_threshold=0.001, predict_threshold=0.5):
    """YOLO için confidence chart"""
    if result.boxes is None or len(result.boxes) == 0:
        return None

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

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(confidences)), confidences, color="steelblue", alpha=0.8)

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

    ax.axhline(
        y=predict_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {predict_threshold}",
    )
    ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


# ============================================================================
# RT-DETR FUNCTIONS
# ============================================================================


def init_distributed_mode():
    """Distributed training'i single process için init et"""
    import torch.distributed as dist
    import os

    if dist.is_available() and not dist.is_initialized():
        # Single process için distributed init
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

        try:
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29500",
                rank=0,
                world_size=1,
            )
            print("Distributed process group initialized (single process)")
        except Exception as e:
            print(f"Warning: Could not initialize process group: {e}")


def setup_rtdetr_inference():
    """RT-DETRv4 inference için gerekli path'leri ayarla"""
    import sys

    # RT-DETRv4 repo path'ini sys.path'e ekle
    if str(RTDETR_REPO) not in sys.path:
        sys.path.insert(0, str(RTDETR_REPO))

    # Distributed mode'u init et
    init_distributed_mode()


def load_rtdetr_model_and_config(config_path, checkpoint_path, device="cpu"):
    """RT-DETRv4 modelini ve config'i yükle (working method)"""
    import torch

    setup_rtdetr_inference()

    try:
        from engine.core import YAMLConfig

        print(f"Loading config from: {config_path}")
        cfg = YAMLConfig(str(config_path))

        print("Building model from config...")
        model = cfg.model

        if model is None:
            raise ValueError("cfg.model is None. Check your config file.")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

        # Checkpoint keys'leri kontrol et
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # State dict yükle
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print("Loaded state dict from checkpoint['model']")
        elif "ema" in checkpoint:
            if isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
                model.load_state_dict(checkpoint["ema"]["module"])
            else:
                model.load_state_dict(checkpoint["ema"])
            print("Loaded state dict from checkpoint['ema']")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint directly")

        model.to(device).eval()
        print(f"Model loaded successfully and moved to {device}")

        return model, cfg

    except Exception as e:
        import traceback

        print(f"Detailed error: {traceback.format_exc()}")
        raise Exception(f"Model loading failed: {str(e)}")


def rtdetr_inference_image(model, image_path, cfg, device="cpu"):
    """
    RT-DETRv4 inference (working method)

    Returns:
        outputs: model raw outputs
        orig_img: original image (BGR)
        orig_size: (height, width)
    """
    import torch

    # Image yükle
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        raise ValueError(f"Image not found: {image_path}")

    orig_h, orig_w = orig_img.shape[:2]

    # Resize size config'ten al (default: 960)
    eval_size = cfg.yaml_cfg.get("eval_spatial_size", [960, 960])
    if isinstance(eval_size, int):
        eval_size = [eval_size, eval_size]
    resize_h, resize_w = eval_size

    # Preprocessing (working method)
    img = cv2.resize(orig_img, (resize_w, resize_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Original size tensor
    orig_size = torch.tensor([[orig_h, orig_w]], device=device)

    # Inference
    with torch.no_grad():
        outputs = model(img, orig_size)

    return outputs, orig_img, (orig_h, orig_w)


def postprocess_rtdetr_outputs(outputs, orig_size, conf_threshold=0.5):
    """
    RT-DETR raw outputs'u parse et (working method)

    Args:
        outputs: model outputs dict with 'pred_boxes' and 'pred_logits'
        orig_size: (height, width) tuple
        conf_threshold: minimum confidence threshold

    Returns:
        dict with boxes, scores, labels
    """
    import torch

    # Outputs'tan predictions'ı al
    pred_logits = outputs["pred_logits"]  # [batch, num_queries, num_classes+1]
    pred_boxes = outputs["pred_boxes"]  # [batch, num_queries, 4]

    # Scores ve labels (son class no-object olduğu için çıkarıyoruz)
    # scores = logits.softmax(-1)[..., :-1].max(-1)[0]
    # labels = logits.softmax(-1)[..., :-1].argmax(-1)
    probs = pred_logits.softmax(-1)[..., :-1]  # [batch, num_queries, num_classes]
    scores, labels = probs.max(-1)  # Her query için en yüksek score ve label

    # Batch'i çıkar
    scores = scores[0].cpu().numpy()
    labels = labels[0].cpu().numpy()
    pred_boxes = pred_boxes[0].cpu().numpy()

    # Confidence threshold uygula
    keep = scores >= conf_threshold
    scores = scores[keep]
    labels = labels[keep]
    pred_boxes = pred_boxes[keep]

    # Bbox'lar zaten normalized format'ta (cx, cy, w, h)
    # Pixel coordinates'e çevir
    orig_h, orig_w = orig_size
    boxes_xyxy = []

    for box in pred_boxes:
        cx, cy, w, h = box
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy.append([x1, y1, x2, y2])

    return {
        "boxes": np.array(boxes_xyxy) if len(boxes_xyxy) > 0 else np.array([]),
        "scores": scores,
        "labels": labels,
    }


def load_gt_bboxes_from_coco(coco_json_path, image_name):
    """COCO format JSON'dan ground truth bbox'ları yükle"""
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Image ID bul
    image_id = None
    for img in coco_data["images"]:
        if img["file_name"] == image_name:
            image_id = img["id"]
            break

    if image_id is None:
        return []

    # Bu image'a ait annotations'ları bul
    gt_bboxes = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] == image_id:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            category_id = ann["category_id"]

            # Category name bul
            category_name = None
            for cat in coco_data["categories"]:
                if cat["id"] == category_id:
                    category_name = cat["name"]
                    break

            gt_bboxes.append(
                {
                    "bbox": [x, y, x + w, y + h],  # [x1, y1, x2, y2]
                    "class": category_id - 1,  # 0-indexed
                    "class_name": category_name,
                }
            )

    return gt_bboxes


def visualize_bboxes_rtdetr(
    img, gt_bboxes, pred_detections, class_names, conf_threshold=0.5
):
    """RT-DETR için bbox görselleştirme"""
    h, w = img.shape[:2]

    # GT IMAGE
    gt_img = img.copy()
    for gt in gt_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in gt["bbox"]]
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = gt["class_name"] if gt["class_name"] else f"Class {gt['class']}"
        cv2.putText(
            gt_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # PRED IMAGE
    pred_img = img.copy()
    for i, (box, score, label_id) in enumerate(
        zip(
            pred_detections["boxes"],
            pred_detections["scores"],
            pred_detections["labels"],
        )
    ):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        class_name = (
            class_names[label_id]
            if class_names and label_id < len(class_names)
            else f"Class {label_id}"
        )
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(
            pred_img,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # OVERLAY (bbox overlap visualization)
    overlay = img.copy()

    # GT bboxes - Green
    for gt in gt_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in gt["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Pred bboxes - Red
    for box, score in zip(pred_detections["boxes"], pred_detections["scores"]):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return gt_img, pred_img, overlay


def plot_confidence_chart_rtdetr(
    detections, class_names, conf_threshold=0.001, predict_threshold=0.5, top_k=10
):
    """RT-DETR için confidence chart"""
    scores = detections["scores"]
    labels = detections["labels"]

    if len(scores) == 0:
        return None

    # Confidence threshold
    keep = scores >= conf_threshold
    scores = scores[keep]
    labels = labels[keep]

    if len(scores) == 0:
        return None

    if top_k is not None and len(scores) > top_k:
        topk_idx = np.argsort(scores)[-top_k:][::-1]
        scores = scores[topk_idx]
        labels = labels[topk_idx]

    # Class names
    class_name_list = [
        f"{class_names[l] if l < len(class_names) else f'Class {l}'}\n({s:.3f})"
        for l, s in zip(labels, scores)
    ]

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(scores)), scores, alpha=0.8)

    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel("Sınıf Adı", fontsize=12, fontweight="bold")
    ax.set_ylabel("Confidence Değeri", fontsize=12, fontweight="bold")
    ax.set_title(
        "RT-DETRv4 Confidence Scores (best 10 in 300 queries)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(class_name_list, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax.axhline(
        y=predict_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {predict_threshold}",
    )
    ax.legend()

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/predict/test")
def predict_test(
    model_name: str,
    image_name: str,
    conf_threshold: float = 0.5,
    chart_conf_threshold: float = 0.001,
):
    """Test prediction endpoint - YOLO ve RT-DETR destekli"""

    image_path = TEST_DIR / "images" / image_name

    if not image_path.exists():
        return {"error": f"image not found: {image_name}"}

    # ========================================================================
    # YOLO MODELS
    # ========================================================================
    if model_name.startswith("yolo"):

        if model_name.startswith("yolov12"):
            chdir("yolov12")

        from ultralytics import YOLO

        model = YOLO(MODELS_DIR / f"{model_name}.pt")
        image_label_path = TEST_DIR / "labels" / f"{image_name.split('.')[0]}.txt"

        if not image_label_path.exists():
            return {"error": f"label file not found for {image_name}"}

        start_time = time.time()
        results = model.predict(
            source=str(image_path), imgsz=1536, device="cpu", conf=chart_conf_threshold
        )
        inference_time = time.time() - start_time

        result = results[0]

        img_gt, img_pred, img_overlay = visualize_gt_pred_overlay(
            image_label_path, result, conf_threshold
        )

        _, gt_buffer = cv2.imencode(".png", img_gt)
        _, pred_buffer = cv2.imencode(".png", img_pred)
        _, overlay_buffer = cv2.imencode(".png", img_overlay)

        confidence_chart = plot_confidence_chart_yolo(
            result,
            conf_threshold=chart_conf_threshold,
            predict_threshold=conf_threshold,
        )

        chdir(BASE_DIR)
        return JSONResponse(
            {
                "gt_image": base64.b64encode(gt_buffer).decode("utf-8"),
                "pred_image": base64.b64encode(pred_buffer).decode("utf-8"),
                "overlay_image": base64.b64encode(overlay_buffer).decode("utf-8"),
                "confidence_chart": confidence_chart,
                "inference_time": inference_time,
                "predictions": [
                    {
                        "class": result.names[int(result.boxes.cls[i])],
                        "confidence": float(result.boxes.conf[i]),
                    }
                    for i in range(len(result.boxes))
                    if float(result.boxes.conf[i]) >= conf_threshold
                ],
            }
        )

    # ========================================================================
    # RT-DETR MODELS (Official Inference Method)
    # ========================================================================
    elif model_name.startswith("rt-detr"):
        config_path = MODELS_DIR / f"{model_name}_config.yml"
        checkpoint_path = MODELS_DIR / f"{model_name}.pth"
        coco_json_path = TEST_DIR / "instances_test.json"

        if not checkpoint_path.exists():
            return {"error": f"checkpoint not found: {checkpoint_path}"}

        if not config_path.exists():
            return {"error": f"config not found: {config_path}"}

        if not coco_json_path.exists():
            return {"error": f"COCO annotation file not found: {coco_json_path}"}

        # Class names'i COCO json'dan al
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
        class_names = [
            cat["name"]
            for cat in sorted(coco_data["categories"], key=lambda x: x["id"])
        ]

        # Model yükle (official method)
        try:
            model, cfg = load_rtdetr_model_and_config(
                config_path, checkpoint_path, device="cpu"
            )
        except Exception as e:
            return {"error": f"Model loading failed: {str(e)}"}

        # Ground truth yükle
        gt_bboxes = load_gt_bboxes_from_coco(coco_json_path, image_name)

        # Inference (official method)
        try:
            start_time = time.time()
            outputs, orig_img, orig_size = rtdetr_inference_image(
                model, image_path, cfg, device="cpu"
            )
            inference_time = time.time() - start_time
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # Postprocess (official method)
        detections = postprocess_rtdetr_outputs(
            outputs, orig_size, conf_threshold=chart_conf_threshold
        )

        # Visualize
        img_gt, img_pred, img_overlay = visualize_bboxes_rtdetr(
            orig_img, gt_bboxes, detections, class_names, conf_threshold
        )

        # Encode images
        _, gt_buffer = cv2.imencode(".png", img_gt)
        _, pred_buffer = cv2.imencode(".png", img_pred)
        _, overlay_buffer = cv2.imencode(".png", img_overlay)

        # Confidence chart
        confidence_chart = plot_confidence_chart_rtdetr(
            detections,
            class_names,
            predict_threshold=conf_threshold,
        )

        return JSONResponse(
            {
                "gt_image": base64.b64encode(gt_buffer).decode("utf-8"),
                "pred_image": base64.b64encode(pred_buffer).decode("utf-8"),
                "overlay_image": base64.b64encode(overlay_buffer).decode("utf-8"),
                "confidence_chart": confidence_chart,
                "inference_time": inference_time,
                "predictions": [
                    {
                        "class": (
                            class_names[label]
                            if label < len(class_names)
                            else f"Class {label}"
                        ),
                        "confidence": float(score),
                    }
                    for score, label in zip(detections["scores"], detections["labels"])
                    if float(score) >= conf_threshold
                ],
            }
        )

    else:
        return {"error": "unsupported model type"}


# ============================================================================
# UPLOAD ENDPOINT
# ============================================================================


@app.post("/predict/upload")
async def predict_upload(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    conf_threshold: float = Form(0.5),
    chart_conf_threshold: float = Form(0.001),
    gt_file: Optional[UploadFile] = File(None),
):
    """
    Upload endpoint - Image + optional GT file

    YOLO: gt_file = .txt (YOLO polygon format)
    RT-DETR: gt_file = .json (simplified COCO format)
    """

    # Image'i memory'de oku
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    gt_error = None  # GT parse hatası varsa buraya kaydedilecek

    # ========================================================================
    # YOLO MODELS
    # ========================================================================
    if model_name.startswith("yolo"):

        if model_name.startswith("yolov12"):
            chdir("yolov12")

        from ultralytics import YOLO

        model = YOLO(MODELS_DIR / f"{model_name}.pt")

        # Inference
        start_time = time.time()
        results = model.predict(
            source=img, imgsz=1536, device="cpu", conf=chart_conf_threshold
        )
        inference_time = time.time() - start_time

        result = results[0]

        # GT varsa parse et
        gt_objs = []
        if gt_file is not None:
            try:
                gt_content = await gt_file.read()
                gt_text = gt_content.decode("utf-8")
                gt_objs = parse_yolo_gt_from_string(gt_text, img.shape, CLASS_NAMES)
            except Exception as e:
                gt_error = f"GT file parse error: {str(e)}"

        # Pred objects
        pred_objs = load_pred_masks(result, conf_threshold)

        # Visualization
        if gt_objs:
            img_gt, img_pred, img_overlay = visualize_gt_pred_overlay_from_objects(
                img, gt_objs, pred_objs, result.names
            )
        else:
            # Sadece prediction
            img_gt = None
            img_pred = visualize_pred_only(img, pred_objs, result.names)
            img_overlay = None

        # Confidence chart
        confidence_chart = plot_confidence_chart_yolo(
            result,
            conf_threshold=chart_conf_threshold,
            predict_threshold=conf_threshold,
        )

        # Response
        response_data = {
            "inference_time": inference_time,
            "confidence_chart": confidence_chart,
            "predictions": [
                {
                    "class": result.names[int(result.boxes.cls[i])],
                    "confidence": float(result.boxes.conf[i]),
                }
                for i in range(len(result.boxes))
                if float(result.boxes.conf[i]) >= conf_threshold
            ],
        }

        # GT hatası varsa ekle
        if gt_error:
            response_data["gt_warning"] = gt_error

        # Images
        if img_gt is not None:
            _, gt_buffer = cv2.imencode(".png", img_gt)
            response_data["gt_image"] = base64.b64encode(gt_buffer).decode("utf-8")

        _, pred_buffer = cv2.imencode(".png", img_pred)
        response_data["pred_image"] = base64.b64encode(pred_buffer).decode("utf-8")

        if img_overlay is not None:
            _, overlay_buffer = cv2.imencode(".png", img_overlay)
            response_data["overlay_image"] = base64.b64encode(overlay_buffer).decode(
                "utf-8"
            )

        chdir(BASE_DIR)
        return JSONResponse(response_data)

    # ========================================================================
    # RT-DETR MODELS
    # ========================================================================
    elif model_name.startswith("rt-detr"):
        config_path = MODELS_DIR / f"{model_name}_config.yml"
        checkpoint_path = MODELS_DIR / f"{model_name}.pth"

        if not checkpoint_path.exists():
            return {"error": f"checkpoint not found: {checkpoint_path}"}

        if not config_path.exists():
            return {"error": f"config not found: {config_path}"}

        # Model yükle
        try:
            model, cfg = load_rtdetr_model_and_config(
                config_path, checkpoint_path, device="cpu"
            )
        except Exception as e:
            return {"error": f"Model loading failed: {str(e)}"}

        # Inference
        try:
            start_time = time.time()
            outputs, orig_img, orig_size = rtdetr_inference_image_from_array(
                model, img, cfg, device="cpu"
            )
            inference_time = time.time() - start_time
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # Postprocess
        detections = postprocess_rtdetr_outputs(
            outputs, orig_size, conf_threshold=conf_threshold
        )

        # GT varsa parse et
        gt_bboxes = []
        if gt_file is not None:
            try:
                gt_content = await gt_file.read()
                gt_json = json.loads(gt_content.decode("utf-8"))
                gt_bboxes = parse_rtdetr_gt_from_json(gt_json, CLASS_NAMES)
            except Exception as e:
                gt_error = f"GT file parse error: {str(e)}"

        # Visualization
        if gt_bboxes:
            img_gt, img_pred, img_overlay = visualize_bboxes_rtdetr(
                orig_img, gt_bboxes, detections, CLASS_NAMES, conf_threshold
            )
        else:
            # Sadece prediction
            img_gt = None
            img_pred = visualize_rtdetr_pred_only(
                orig_img, detections, CLASS_NAMES, conf_threshold
            )
            img_overlay = None

        # Confidence chart
        confidence_chart = plot_confidence_chart_rtdetr(
            detections,
            CLASS_NAMES,
            predict_threshold=conf_threshold,
        )

        # Response
        response_data = {
            "inference_time": inference_time,
            "confidence_chart": confidence_chart,
            "predictions": [
                {
                    "class": (
                        CLASS_NAMES[label]
                        if label < len(CLASS_NAMES)
                        else f"Class {label}"
                    ),
                    "confidence": float(score),
                }
                for score, label in zip(detections["scores"], detections["labels"])
                if float(score) >= conf_threshold
            ],
        }

        # GT hatası varsa ekle
        if gt_error:
            response_data["gt_warning"] = gt_error

        # Images
        if img_gt is not None:
            _, gt_buffer = cv2.imencode(".png", img_gt)
            response_data["gt_image"] = base64.b64encode(gt_buffer).decode("utf-8")

        _, pred_buffer = cv2.imencode(".png", img_pred)
        response_data["pred_image"] = base64.b64encode(pred_buffer).decode("utf-8")

        if img_overlay is not None:
            _, overlay_buffer = cv2.imencode(".png", img_overlay)
            response_data["overlay_image"] = base64.b64encode(overlay_buffer).decode(
                "utf-8"
            )

        return JSONResponse(response_data)

    else:
        return {"error": "unsupported model type"}


@app.get("/get/models")
def get_models():
    models = []
    for model_file in MODELS_DIR.iterdir():
        if (
            model_file.is_file()
            and (model_file.name.endswith(("pt", "pth")))
            and not model_file.stem.endswith("config")
        ):
            models.append(model_file.stem)

    return models


@app.get("/get/test-images")
def get_test_images():
    return [i.name for i in (TEST_DIR / "images").iterdir() if i.is_file()]


@app.get("/get/models-info")
def get_models_info(model_name: str):
    model_dir = RESULTS_DIR / model_name

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # ---- model info ----
    model_info_file = model_dir / "model_info.json"
    metrics_file = model_dir / "metrics_test_classwise.json"

    model_info = {}
    metrics = {}

    if model_info_file.exists():
        model_info = json.loads(model_info_file.read_text())

    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())

    graphics = []

    if model_dir.exists():
        graphics = [
            {
                "label": img.stem.replace("_", " ").title(),
                "path": f"/results/{model_name}/{img.name}",
            }
            for img in sorted(model_dir.glob("*.png"))
        ]

    return {
        "model": model_name,
        "model_info": model_info,
        "metrics": metrics,
        "graphics": graphics,
    }
