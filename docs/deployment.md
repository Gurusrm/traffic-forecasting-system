# Deployment Guide

## Docker
1.  **Build**: `docker-compose build`
2.  **Run**: `docker-compose up`
3.  **Access**: `http://localhost:8000`

## Manual Deployment
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run server: `python src/inference/serve.py`
3.  Set `PORT` environment variable to configure the listening port.

## Environment Variables
-   `CUDA_VISIBLE_DEVICES`: Select GPU ID (default: 0)
-   `CONFIG_PATH`: Path to custom config YAML (optional)
