@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem CROP-AI training / dataset entrypoint (cmd.exe and PowerShell).
rem Usage: train_windows.bat [hardware|dataset|train|all]

cd /d "%~dp0"
set "CROP_AI_ROOT=%CD%"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "STAGE=%~1"
if "%STAGE%"=="" set "STAGE=all"

set "PYTHON_BIN="
if exist "%CD%\.venv\Scripts\python.exe" set "PYTHON_BIN=%CD%\.venv\Scripts\python.exe"
if "%PYTHON_BIN%"=="" (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_BIN=py -3"
  )
)
if "%PYTHON_BIN%"=="" (
  where python >nul 2>&1
  if not errorlevel 1 set "PYTHON_BIN=python"
)
if "%PYTHON_BIN%"=="" (
  echo [FAIL] Python 3 not found. Install Python 3.10+ or create .venv.
  exit /b 1
)

echo CROP-AI SIH26131
echo root: %CD%
echo stage: %STAGE%
%PYTHON_BIN% --version

if /I "%STAGE%"=="hardware" goto HARDWARE
if /I "%STAGE%"=="dataset" goto DATASET
if /I "%STAGE%"=="train" goto TRAIN
if /I "%STAGE%"=="all" goto DATASET
echo Unknown stage: %STAGE%
echo Usage: train_windows.bat [hardware^|dataset^|train^|all]
exit /b 2

:HARDWARE
%PYTHON_BIN% "%CD%\scripts\check_env.py"
if errorlevel 1 exit /b 1
goto DONE

:DATASET
%PYTHON_BIN% "%CD%\scripts\check_env.py"
if errorlevel 1 (
  echo [FAIL] environment check failed
  exit /b 1
)
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\splits" mkdir "data\splits"
if not exist "data\synthetic" mkdir "data\synthetic"
if not exist "data\manifests" mkdir "data\manifests"
if not exist "runs\detector" mkdir "runs\detector"
if not exist "runs\classifier" mkdir "runs\classifier"
if not exist "runs\segmentation" mkdir "runs\segmentation"
if not exist "runs\risk" mkdir "runs\risk"
if not exist "runs\fusion" mkdir "runs\fusion"
if not exist "runs\deployment" mkdir "runs\deployment"
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"
echo === dataset preparation ===
%PYTHON_BIN% -m cropai prepare-dataset
if errorlevel 1 (
  echo [FAIL] dataset preparation failed
  exit /b 1
)
echo [OK] dataset preparation finished
if /I "%STAGE%"=="all" goto TRAIN
goto DONE

:TRAIN
if not exist "%CD%\src\cropai\vision\train.py" (
  echo [FAIL] src\cropai\vision\train.py missing
  exit /b 1
)
set "LIGHT=0"
echo %EXTRA% | findstr /I /C:"--dry-run" /C:"--smoke" >nul
if not errorlevel 1 set "LIGHT=1"
%PYTHON_BIN% -c "from cropai.utils.hardware import detect_hardware; import sys; sys.exit(0 if detect_hardware().get('cuda_available') else 3)"
if errorlevel 3 (
  echo WARNING:
  echo CUDA unavailable.
  echo.
  echo Continuing with CPU fallback where practical.
  if not "%LIGHT%"=="1" if not "%CROP_AI_ALLOW_CPU_TRAIN%"=="1" (
    echo Refusing heavy training without CUDA. Use --dry-run / --smoke or set CROP_AI_ALLOW_CPU_TRAIN=1.
    exit /b 2
  )
)
%PYTHON_BIN% "%CD%\src\cropai\vision\train.py" %EXTRA%
if errorlevel 1 exit /b 1
goto DONE

:DONE
echo [OK] stage '%STAGE%' complete
exit /b 0
