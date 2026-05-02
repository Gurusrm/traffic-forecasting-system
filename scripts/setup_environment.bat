@echo off
echo Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo Setup complete. activate with: venv\Scripts\activate
