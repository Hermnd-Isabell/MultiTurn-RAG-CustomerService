@echo off
setlocal
echo Running RAG Test Suite...

:: Set paths to use venv
set "VENV_DIR=%~dp0venv"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

:: Handle potential library conflicts (same as run_rag.bat)
set KMP_DUPLICATE_LIB_OK=TRUE

:: Disable HuggingFace hub network check (same as run_rag.bat)
set HF_HUB_OFFLINE=1

:: Check if pytest is installed
python -c "import pytest" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] pytest not found.
    echo Please install test dependencies first:
    echo   .\venv\Scripts\pip install -r requirements-dev.txt
    echo.
    pause
    exit /b 1
)

:: Run tests
echo.
python -m pytest tests -v --tb=short

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Tests finished with failures.
    pause
) else (
    echo.
    echo All tests passed.
    pause
)
