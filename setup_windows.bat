@echo off
REM ==========================================================================
REM  Credit Card Fraud Detection - one-click setup for Windows
REM  1. creates a virtual environment in .\venv
REM  2. installs the libraries from requirements.txt
REM  3. generates the dataset, draws the charts and trains the models
REM ==========================================================================
setlocal
cd /d "%~dp0"

echo.
echo ==========================================================
echo   Credit Card Fraud Detection - Windows setup
echo ==========================================================
echo.

REM ---- find Python: "python" on PATH, otherwise the "py" launcher --------------
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY (
    py -3 --version >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    echo [ERROR] Python is not installed or not on PATH.
    echo         Install Python 3.12, 3.13 or 3.14 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" on the first installer screen.
    echo         Then close this window and run setup_windows.bat again.
    pause
    exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required. Installed version:
    %PY% --version
    pause
    exit /b 1
)
echo Using:
%PY% --version
echo.

REM ---- 1. virtual environment -------------------------------------------------
if not exist "venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment in .\venv ...
    %PY% -m venv venv
    if errorlevel 1 goto :fail
) else (
    echo [1/4] Virtual environment already exists - reusing it.
)

REM ---- 2 + 3. libraries -----------------------------------------------------------
echo.
echo [2/4] Upgrading pip ...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo [3/4] Installing libraries - this can take 3 to 5 minutes ...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

REM ---- 4. data, charts, models ----------------------------------------------------
echo.
echo [4/4] Generating data, drawing charts and training models ...
"venv\Scripts\python.exe" main.py
if errorlevel 1 goto :fail

echo.
echo ==========================================================
echo   Setup complete!
echo     Charts : reports\figures
echo     Models : models
echo   Double-click run_app.bat to open the dashboard.
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed - read the messages above.
echo         The Troubleshooting section of README.md lists common fixes.
pause
exit /b 1
