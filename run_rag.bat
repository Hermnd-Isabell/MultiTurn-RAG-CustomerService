@echo off
setlocal
echo Starting RAG Application...

:: Set paths to use venv
set "VENV_DIR=%~dp0venv"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

:: Set environment variables to handle potential conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

:: Disable HuggingFace hub network check to avoid timeout on model load
:: (sentence-transformers tries to fetch adapter_config.json on import)
set HF_HUB_OFFLINE=1

:: Run the application
python pkg/webrun.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application crashed with error code %ERRORLEVEL%.
    pause
)
pause
