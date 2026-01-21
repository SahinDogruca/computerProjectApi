<div align="center">

# 🦷 Dental Lesion Detection API

**An intelligent REST API for detecting and classifying oral lesions in dental X-ray images using state-of-the-art deep learning models.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v12-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com/)

---

<img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status"/>
<img src="https://img.shields.io/badge/License-Academic-blue?style=flat-square" alt="License"/>

</div>

---

## 📋 Overview

This backend API provides powerful computer vision capabilities for dental pathology detection. It supports multiple deep learning architectures including **YOLOv8**, **YOLOv12**, and **RT-DETR** for both object detection and instance segmentation tasks.

### 🎯 Supported Lesion Classes

| Class | Description |
|-------|-------------|
| 🦴 **Dentigerous Cyst** | Developmental odontogenic cyst |
| 🔬 **Keratocyst** | Odontogenic keratocyst (formerly OKC) |
| 🔴 **Radicular Cyst** | Inflammatory periapical cyst |
| 🧬 **Ameloblastoma** | Benign odontogenic tumor |
| 🦷 **Odontoma** | Benign mixed odontogenic tumor |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Core Capabilities
- **Multi-Model Support** — YOLO (v8, v12) & RT-DETR architectures
- **Dual Mode** — Object detection & instance segmentation
- **Real-time Inference** — Optimized CPU/GPU prediction
- **Confidence Visualization** — Interactive charts for predictions

</td>
<td width="50%">

### 🛠️ Developer Features
- **RESTful API** — Clean endpoints with FastAPI
- **Model Upload** — Dynamic model loading system
- **Ground Truth Comparison** — GT vs Prediction overlays
- **YOLO Label Support** — Native bbox & polygon formats

</td>
</tr>
</table>

---

## 🏗️ Project Structure

```
backend/
├── 📜 api.py              # FastAPI application & endpoints
├── 🧠 models.py           # Model manager (YOLO, RT-DETR)
├── 🎨 visualization.py    # Detection visualization & overlays
├── 🔧 utils.py            # Utilities & chart generation
├── 📦 requirements.txt    # Python dependencies
│
├── 📁 models/             # Pre-trained model weights (.pt)
│   ├── yolov8x-seg.pt
│   ├── yolov12l-seg.pt
│   └── rt-detrv4x.pt
│
├── 📁 data/
│   └── test/              # Test images & labels
│       ├── images/
│       └── labels/
│
├── 📁 results/            # Model metrics & visualizations
└── 📁 yolov12/            # YOLOv12 custom implementation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the development server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

---

## 🐳 Docker Deployment

For consistent deployment across different machines, use Docker:

### Quick Start with Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the container
docker-compose down
```

### Manual Docker Build

```bash
# Build the image
docker build -t lesion-detection-api .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  lesion-detection-api
```

### Docker Files

| File | Description |
|------|-------------|
| `Dockerfile` | Main container definition (CPU-optimized) |
| `docker-compose.yml` | Orchestration with volume mounts |
| `requirements-docker.txt` | Pinned dependencies for Docker |
| `.dockerignore` | Files excluded from build context |

> **Note:** The Docker image uses CPU-only PyTorch (~2GB smaller than GPU version). Model weights are mounted as volumes for easy updates.

---

## 📚 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

<details>
<summary><b>🟢 GET /</b> — Health Check</summary>

Returns the API status.

**Response:**
```json
{
  "status": "app is running"
}
```
</details>

<details>
<summary><b>🟢 GET /predict/test</b> — Predict on Test Image</summary>

Run inference on a pre-loaded test image with ground truth comparison.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | string | ✅ | - | Model to use (e.g., `yolov8x-seg`) |
| `image_name` | string | ✅ | - | Test image filename |
| `conf_threshold` | float | ❌ | 0.5 | Confidence threshold for display |
| `chart_conf_threshold` | float | ❌ | 0.001 | Min confidence for chart |

**Response:**
```json
{
  "gt_image": "base64...",
  "pred_image": "base64...",
  "overlay_image": "base64...",
  "confidence_chart": "base64...",
  "inference_time": 1.234,
  "model_type": "segmentation",
  "total_detections": 3
}
```
</details>

<details>
<summary><b>🔵 POST /predict/upload</b> — Predict on Uploaded Image</summary>

Upload an image for inference with optional ground truth file.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✅ | Image file to analyze |
| `model_name` | string | ✅ | Model to use |
| `conf_threshold` | float | ❌ | Confidence threshold (default: 0.5) |
| `chart_conf_threshold` | float | ❌ | Chart threshold (default: 0.001) |
| `top_k` | int | ❌ | Top K predictions to show (default: 10) |
| `gt_file` | file | ❌ | Ground truth label file (YOLO format) |

**Response:**
```json
{
  "pred_image": "base64...",
  "confidence_chart": "base64...",
  "inference_time": 0.892,
  "model_type": "detection",
  "total_predictions": 5,
  "filtered_predictions_count": 2
}
```
</details>

<details>
<summary><b>🟢 GET /get/models</b> — List Available Models</summary>

Returns list of all loaded model names.

**Response:**
```json
["yolov8x-seg", "yolov12l-seg", "rt-detrv4x"]
```
</details>

<details>
<summary><b>🟢 GET /get/test-images</b> — List Test Images</summary>

Returns list of test image filenames.

**Response:**
```json
["image001.png", "image002.png", "image003.png"]
```
</details>

<details>
<summary><b>🟢 GET /get/models-info</b> — Get Model Metrics</summary>

Get detailed model information and performance metrics.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | ✅ | Model name to query |

**Response:**
```json
{
  "model": "yolov8x-seg",
  "model_info": { ... },
  "metrics": { ... },
  "graphics": [
    { "label": "Confusion Matrix", "path": "/results/..." }
  ]
}
```
</details>

<details>
<summary><b>🔵 POST /load/model</b> — Upload New Model</summary>

Upload a new trained model file.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Model file (.pt, .pth, .yml) |
| `model_name` | string | ✅ | Name for the model (e.g., `yolov8x-custom`) |

**Response:**
```json
{
  "message": "Model uploaded successfully",
  "filename": "yolov8x-custom"
}
```
</details>

---

## 🧠 Supported Models

| Model | Type | Architecture | Segmentation |
|-------|------|--------------|--------------|
| `yolov8x-seg` | Detection + Segmentation | YOLOv8 | ✅ |
| `yolov8x-seg-pretrained` | Detection + Segmentation | YOLOv8 | ✅ |
| `yolov12l-seg` | Detection + Segmentation | YOLOv12 | ✅ |
| `yolov12l-seg-pretrained` | Detection + Segmentation | YOLOv12 | ✅ |
| `rt-detrv4x` | Detection | RT-DETR | ❌ |
| `rt-detrv4x-pretrained` | Detection | RT-DETR | ❌ |

---

## 📊 Visualization Features

The API provides rich visualization capabilities:

- **Ground Truth Image** — Original labels with bounding boxes/masks (green)
- **Prediction Image** — Model predictions with confidence scores (red)
- **Overlay Image** — Combined GT and predictions for comparison
- **Confidence Chart** — Bar chart showing top-K predictions with threshold line

---

## 🔧 Configuration

### CORS Settings

The API is configured for local development with frontend at `http://localhost:5173`. Modify in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Model Inference Settings

Default inference configuration in `models.py`:
- **Image Size:** 1536px
- **Device:** CPU (change to `cuda` for GPU)
- **Confidence Threshold:** 0.001 (captures all detections)

---

## 🔬 Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **Deep Learning** | PyTorch 2.7.0, Ultralytics |
| **Computer Vision** | OpenCV, Supervision |
| **Visualization** | Matplotlib |
| **Validation** | faster-coco-eval |

---

## 📝 YOLO Label Format

The API supports standard YOLO label formats:

**Bounding Box Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```

**Segmentation Format:**
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

All coordinates are normalized (0-1 range).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is developed for academic purposes as part of a university semester project.

---

<div align="center">

**Şahin Doğruca | Kıvanç Erdem Sarıkamış | Hamza Osman İlhan**

*Yıldız Technical University — Computer Engineering*

</div>
