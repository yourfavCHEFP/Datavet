@echo off
title DataVet Pro Streamlit - Port 8501
echo.
echo  =============================================
echo    DataVet Pro  ^|  http://localhost:8501
echo  =============================================
echo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
	echo [Setup] Creating virtual environment...
	py -3 -m venv .venv
)

echo [Setup] Ensuring pip is available...
.venv\Scripts\python.exe -m ensurepip --upgrade >nul 2>&1

echo [Setup] Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt

for /f "tokens=5" %%P in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
	echo [Setup] Releasing port 8501 from PID %%P...
	taskkill /PID %%P /F >nul 2>&1
)

echo [Run] Starting Streamlit app...
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address localhost --server.port 8501
pause
