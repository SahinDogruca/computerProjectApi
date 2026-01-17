from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass
import json
import shutil


from models import ModelManager
from visualization import Visualizer
from utils import read_yolo_labels, create_confidence_chart

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

CLASS_NAMES = [
    "dentigeroz kist",
    "keratokist",
    "radikuler kist",
    "ameloblastoma",
    "odontoma",
]


@dataclass
class PredictionConfig:
    model_name: str
    conf_threshold: float = 0.5
    chart_conf_threshold: float = 0.001
    top_k: int = 10


app = FastAPI(title="predictApi", version="1.0.0")


app.mount("/results", StaticFiles(directory="results"), name="results")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager(MODELS_DIR)
visualizer = Visualizer()


@app.get("/")
def root():
    return {"status": "app is running"}


@app.get("/predict/test")
def predict_test(
    model_name: str,
    image_name: str,
    conf_threshold: float = 0.5,
    chart_conf_threshold: float = 0.001,
):

    config = PredictionConfig(
        model_name=model_name,
        conf_threshold=conf_threshold,
        chart_conf_threshold=chart_conf_threshold,
    )

    # Validate paths
    image_path = TEST_DIR / "images" / image_name
    label_path = TEST_DIR / "labels" / f"{image_name.split('.')[0]}.txt"

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")

    if not label_path.exists():
        raise HTTPException(status_code=404, detail=f"Label not found for {image_name}")

    model, is_segmentation = model_manager.load_model(config.model_name)
    results, inference_time = model_manager.predict(
        model, str(image_path), config.chart_conf_threshold
    )

    all_detections = model_manager.get_detections(results[0])

    images = visualizer.create_gt_pred_overlay(
        image_path=image_path,
        label_path=label_path,
        predictions=all_detections,
        class_names=results[0].names,
        conf_threshold=config.conf_threshold,
        is_segmentation=is_segmentation,
    )

    confidence_chart = create_confidence_chart(
        detections=all_detections,
        class_names=results[0].names,
        conf_threshold=config.conf_threshold,
        chart_conf_threshold=config.chart_conf_threshold,
    )

    filtered_detections = all_detections[
        all_detections.confidence >= config.conf_threshold
    ]

    return JSONResponse(
        {
            "gt_image": images["gt_image"],
            "pred_image": images["pred_image"],
            "overlay_image": images["overlay_image"],
            "confidence_chart": confidence_chart,
            "inference_time": inference_time,
            "model_type": "segmentation" if is_segmentation else "detection",
            "total_detections": len(filtered_detections),
        }
    )


@app.post("/predict/upload")
async def predict_upload(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    conf_threshold: float = Form(0.5),
    chart_conf_threshold: float = Form(0.001),
    top_k: int = Form(10),
    gt_file: Optional[UploadFile] = File(None),
):

    config = PredictionConfig(
        model_name=model_name,
        conf_threshold=conf_threshold,
        chart_conf_threshold=chart_conf_threshold,
        top_k=top_k,
    )

    img = await model_manager.read_uploaded_image(image)

    model, is_segmentation = model_manager.load_model(config.model_name)
    results, inference_time = model_manager.predict(
        model, img, config.chart_conf_threshold
    )

    all_detections = model_manager.get_detections(results[0])

    gt_data = None
    if gt_file:
        gt_data = await visualizer.process_gt_file(gt_file, img.shape)

    if gt_data and gt_data.get("detections"):
        images = visualizer.create_gt_pred_overlay_from_detections(
            img=img,
            gt_detections=gt_data["detections"],
            pred_detections=all_detections,
            class_names=results[0].names,
            conf_threshold=config.conf_threshold,
            is_segmentation=is_segmentation,
            gt_is_seg=gt_data["is_segmentation"],
        )
    else:
        images = visualizer.create_pred_only(
            img=img,
            predictions=all_detections,
            class_names=results[0].names,
            conf_threshold=config.conf_threshold,
            is_segmentation=is_segmentation,
        )

    confidence_chart = create_confidence_chart(
        detections=all_detections,
        class_names=results[0].names,
        conf_threshold=config.conf_threshold,
        chart_conf_threshold=config.chart_conf_threshold,
        top_k=config.top_k,
    )

    filtered_detections = all_detections[
        all_detections.confidence >= config.conf_threshold
    ]

    response = {
        "inference_time": inference_time,
        "confidence_chart": confidence_chart,
        "model_type": "segmentation" if is_segmentation else "detection",
        "total_predictions": len(all_detections),
        "filtered_predictions_count": len(filtered_detections),
    }

    # Add GT warning if there was an error
    if gt_data and gt_data.get("error"):
        response["gt_warning"] = gt_data["error"]

    # Add images
    response.update(images)

    return JSONResponse(response)


@app.get("/get/models")
def get_models():
    """Get list of available models"""
    return model_manager.list_models()


@app.get("/get/test-images")
def get_test_images():
    """Get list of test images"""
    images_dir = TEST_DIR / "images"
    return [img.name for img in images_dir.iterdir() if img.is_file()]


@app.get("/get/models-info")
def get_models_info(model_name: str):
    """Get model information and metrics"""
    model_dir = RESULTS_DIR / model_name

    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # Load model info
    model_info_file = model_dir / "model_info.json"
    metrics_file = model_dir / "metrics_test_classwise.json"

    model_info = {}
    metrics = {}

    if model_info_file.exists():
        model_info = json.loads(model_info_file.read_text())

    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())

    # Get graphics
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


@app.post("/load/model")
async def load_model(file: UploadFile = File(...), model_name: str = Form(...)):
    """Upload a new model"""

    # Validate file extension
    if not file.filename.endswith((".pt", ".pth", ".yml")):
        raise HTTPException(
            status_code=400, detail="Only .pt, .pth, .yml files are allowed"
        )

    # Validate model name
    if not model_name.startswith(("yolo", "rt-detr")):
        raise HTTPException(
            status_code=400, detail="Only Yolo or Rt-detr models allowed"
        )

    if model_name.endswith((".pt", ".pth", ".yml")):
        raise HTTPException(status_code=400, detail="Enter model name without suffix")

    # Create safe filename
    suffix = file.filename.split(".")[-1]
    safe_filename = f"{model_name.replace(' ', '_')}.{suffix}"

    # Check if model already exists
    if safe_filename in [m.name for m in MODELS_DIR.iterdir()]:
        raise HTTPException(status_code=400, detail="Same model already exists")

    # Save file
    file_path = MODELS_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": "Model uploaded successfully",
            "filename": safe_filename.split(".")[0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
