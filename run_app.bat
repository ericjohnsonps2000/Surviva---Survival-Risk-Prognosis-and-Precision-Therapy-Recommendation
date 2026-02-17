@echo off
set "SCRIPTDIR=%~dp0"
if "%SCRIPTDIR:~-1%"=="\" set "SCRIPTDIR=%SCRIPTDIR:~0,-1%"
cd /d "%SCRIPTDIR%"
echo Starting Survival Risk app...
echo.
echo When you see "Local URL: http://localhost:8501", open that link in Chrome.
echo Keep this window open while using the app. Close the window to stop the app.
echo.
if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv not found. Run run_training.bat first to create it.
    pause
    exit /b 1
)
if not exist "cox_model.pkl" (
    echo ERROR: cox_model.pkl not found. Run run_training.bat first to train models.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m streamlit run app.py
pause
