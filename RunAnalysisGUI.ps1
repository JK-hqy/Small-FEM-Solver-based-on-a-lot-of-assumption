# 3-Legged OSP Jacket Structural Analysis
# PowerShell Launch Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   3-Legged OSP Jacket Structural Analysis" -ForegroundColor Cyan
Write-Host "   Interactive GUI Version 7" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting application..." -ForegroundColor Yellow
Write-Host "(Dependencies will be installed automatically if needed)" -ForegroundColor Gray
Write-Host ""

# Get the directory where this script is located
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Try to run with python
try {
    python JacketAnalysisGUI.py
}
catch {
    Write-Host "Python command failed, trying py command..." -ForegroundColor Yellow
    try {
        py JacketAnalysisGUI.py
    }
    catch {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "ERROR: Could not run the application." -ForegroundColor Red
        Write-Host "Please make sure Python is installed and in your PATH." -ForegroundColor Red
        Write-Host ""
        Write-Host "You can install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

