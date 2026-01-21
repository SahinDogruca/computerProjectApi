# =============================================================================
# Dental Lesion Detection API - Dockerfile
# CPU-based inference environment
# =============================================================================

# Use Python 3.11 slim image (compatible with all dependencies)
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    # Build tools (for some pip packages)
    gcc \
    g++ \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-docker.txt

# Copy YOLOv12 custom implementation first
COPY yolov12/ ./yolov12/

# Install YOLOv12 as editable package (needed for custom ultralytics)
RUN cd yolov12 && pip install -e . --no-deps

# Copy application code
COPY api.py .
COPY models.py .
COPY visualization.py .
COPY utils.py .

# Create necessary directories
RUN mkdir -p data/test/images data/test/labels models results

# Copy models directory (weights)
# Note: For production, consider mounting this as a volume instead
COPY models/ ./models/

# Copy test data if needed
COPY data/ ./data/

# Copy results/metrics if needed
COPY results/ ./results/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
