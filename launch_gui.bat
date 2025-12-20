@echo off
echo ========================================
echo  Molecular Design System - Web GUI
echo ========================================
echo.
echo Activating energetic_env conda environment...
call conda activate energetic_env

echo.
echo Starting Flask web server...
echo Navigate to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

cd gui
python app.py
