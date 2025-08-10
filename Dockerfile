FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# System deps (minimal) for scientific Python/TensorFlow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install runtime Python deps (CPU TensorFlow for Linux)
RUN pip install --no-cache-dir \
    fastapi==0.111.* \
    uvicorn[standard]==0.30.* \
    jinja2==3.1.* \
    pydantic==2.7.* \
    pandas==2.2.* \
    numpy==1.26.* \
    scikit-learn==1.5.* \
    tensorflow==2.15.0 \
    keras==2.15.0 \
    joblib==1.3.* \
    python-multipart==0.0.7

# Copy application code and artifacts
COPY app ./app
COPY static ./static
COPY templates ./templates
COPY artifacts ./artifacts
COPY main.py ./main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]


