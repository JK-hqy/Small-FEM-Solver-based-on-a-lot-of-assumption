@echo off
title Jacket Structural Analysis - Customizable Geometry v8
echo ============================================================
echo    Jacket Structural Analysis - Customizable Geometry
echo    Version 8
echo ============================================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
python JacketAnalysisGUI_v2.py
if %errorlevel% neq 0 (
    py JacketAnalysisGUI_v2.py
)
pause

