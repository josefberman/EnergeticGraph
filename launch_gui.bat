@echo off
cls
echo.
echo    Activating conda environment: energetic_env
call conda activate energetic_env_py311

echo    Starting EMDS Web Server...
echo.

cd gui
python app.py
