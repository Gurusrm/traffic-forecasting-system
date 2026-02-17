@echo off
echo ====================================================
echo      TRAFFIC AI COMMAND CENTER - GPU LAUNCHER
echo ====================================================
echo.
echo Checking for Python 3.13...
py -3.13 --version
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.13 not found! Please install Python 3.13.
    pause
    exit /b
)

echo.
echo Launching Dashboard on Trichy Map...
echo.
py -3.13 -m streamlit run app.py
pause
