from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"


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
