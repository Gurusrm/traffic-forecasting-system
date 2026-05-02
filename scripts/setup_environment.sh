#!/bin/bash
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Setup complete."
