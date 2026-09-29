@echo off
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python -m pip install -r requirements.txt --disable-pip-version-check -q
python main.py
if errorlevel 1 pause
