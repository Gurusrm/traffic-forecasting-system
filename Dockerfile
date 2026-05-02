# Base image with Python 3.10 (compatible with PyG)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch and PyG installation might need specific index URLs for CPU/CUDA versions in Docker
# Here we'll default to CPU for the standard container to ensure runnability everywhere
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "src/inference/serve.py", "--host", "0.0.0.0", "--port", "8000"]
