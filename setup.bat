@echo off
REM setup.bat - one-shot setup + launch for relational-reality (Windows).
REM Mirrors setup.sh. Double-click it in Explorer, or run from a terminal:
REM
REM     setup.bat
REM
REM Safe to re-run; it reuses the existing .venv and just re-checks/updates.

setlocal enableextensions
cd /d "%~dp0"

REM --- Find a Python launcher: prefer the 'py' launcher, then 'python'. -------
set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH (
    where python >nul 2>nul && set "PYLAUNCH=python"
)
if not defined PYLAUNCH (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then re-run setup.bat.
    pause
    exit /b 1
)

REM --- Require Python 3.11+ (the config loader uses the stdlib tomllib). ------
%PYLAUNCH% -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11+ is required. Found:
    %PYLAUNCH% --version
    echo Install a newer Python from https://www.python.org/downloads/ and re-run.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PYLAUNCH% --version 2^>^&1') do echo ==^> Using %%v

REM --- Create the venv if it doesn't exist yet. -------------------------------
if not exist ".venv" (
    echo ==^> Creating virtual environment in .venv\
    %PYLAUNCH% -m venv .venv
    if errorlevel 1 (
        echo ERROR: failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo ==^> Reusing existing .venv\
)

set "VENV_PY=.venv\Scripts\python.exe"

echo ==^> Upgrading pip
"%VENV_PY%" -m pip install --upgrade pip >nul

echo ==^> Installing dependencies from requirements.txt
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed. See the messages above.
    echo If numba failed to install, check that your Python version has a
    echo matching numba/llvmlite wheel ^(very new Python releases can lag^).
    pause
    exit /b 1
)

echo.
echo ==^> Setup complete - launching the dashboard now.
echo     ^(opens in your browser and starts sweeping; press Ctrl-C to stop^)
echo.
"%VENV_PY%" main.py
echo.
echo ==^> Stopped. Re-run setup.bat any time to resume ^(finished cells are skipped^).
pause

endlocal
