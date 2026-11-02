@echo off
REM ==========================================================================
REM  Starts the fraud detection dashboard at http://localhost:8501
REM  Run setup_windows.bat once before using this.
REM ==========================================================================
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist "models\random_forest.pkl" (
    echo Trained models not found - training them now ...
    "venv\Scripts\python.exe" main.py
    if errorlevel 1 goto :fail
)

echo.
echo Starting the dashboard - your browser will open at http://localhost:8501
echo   * Keep this window open while you use the dashboard.
echo   * If it asks for an email address, just press Enter.
echo   * Press Ctrl+C in this window to stop the dashboard.
echo.
"venv\Scripts\python.exe" -m streamlit run app.py
pause
exit /b 0

:fail
echo.
echo [ERROR] Could not train the models - read the messages above.
pause
exit /b 1
