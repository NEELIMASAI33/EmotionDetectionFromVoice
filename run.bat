@echo off
REM Quick start script for Mood Detection Application

echo.
echo ===============================================
echo   Mood Detection Application - Quick Start
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg is not installed or not in PATH
    echo Audio conversion won't work without FFmpeg
    echo Install from: https://ffmpeg.org/download.html
    echo.
)

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ===============================================
echo Starting Flask Backend Server...
echo ===============================================
echo.

cd backend
python app.py
