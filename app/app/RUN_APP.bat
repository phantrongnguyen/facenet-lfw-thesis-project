@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo FACENET DACN - QUICK START
echo ========================================

set "PYTHON_CMD="
where py >nul 2>nul && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>nul && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] Khong tim thay Python.
    echo Hay cai Python 3.10+ va tick "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Tao moi truong ao .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Khong tao duoc virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Da co .venv, bo qua tao moi truong.
)

echo [2/4] Nang cap pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Loi nang cap pip.
    pause
    exit /b 1
)

echo [3/4] Cai dependencies (co the mat vai phut) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Loi cai dat thu vien tu requirements.txt.
    pause
    exit /b 1
)

echo [4/4] Chay app desktop ...
".venv\Scripts\python.exe" app\app_desktop.py

echo.
echo App da dong.
pause
