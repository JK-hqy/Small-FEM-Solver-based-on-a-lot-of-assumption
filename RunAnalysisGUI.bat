@echo off
title 3-Legged OSP Jacket Structural Analysis
echo ============================================================
echo    3-Legged OSP Jacket Structural Analysis
echo    Interactive GUI Version 7
echo ============================================================
echo.
echo Starting application...
echo (Dependencies will be installed automatically if needed)
echo.

REM Try to run with python command
python JacketAnalysisGUI.py
if %errorlevel% neq 0 (
    echo.
    echo Python command failed, trying py command...
    py JacketAnalysisGUI.py
)

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Could not run the application.
    echo Please make sure Python is installed and in your PATH.
    echo.
    echo You can install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo ============================================================
)

pause

