@echo off
set "SCRIPTDIR=%~dp0"
if "%SCRIPTDIR:~-1%"=="\" set "SCRIPTDIR=%SCRIPTDIR:~0,-1%"
cd /d "%SCRIPTDIR%"
if not exist "venv\Scripts\python.exe" (
    echo Creating venv...
    py -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)
venv\Scripts\python.exe train_survival_models.py
pause
