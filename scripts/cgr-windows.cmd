@echo off
setlocal
if not defined CGR_WINDOWS_VENV set "CGR_WINDOWS_VENV=D:\cav-recovery-venvs\metadrive"
if not defined CGR_CACHE_ROOT set "CGR_CACHE_ROOT=D:\cav-recovery-cache"
if not defined CGR_OUTPUT_ROOT set "CGR_OUTPUT_ROOT=D:\cav-recovery-data\windows-runs"
set "PIP_CACHE_DIR=%CGR_CACHE_ROOT%\pip"
set "MPLCONFIGDIR=%CGR_CACHE_ROOT%\matplotlib"
set "IPYTHONDIR=%CGR_CACHE_ROOT%\ipython"
set "XDG_CACHE_HOME=%CGR_CACHE_ROOT%"
if not defined CGR_TEMP_ROOT set "CGR_TEMP_ROOT=D:\cav-recovery-temp"
set "TEMP=%CGR_TEMP_ROOT%"
set "TMP=%TEMP%"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPYCACHEPREFIX=%CGR_CACHE_ROOT%\pycache"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PATH=%CGR_WINDOWS_VENV%\Scripts;%PATH%"
"%CGR_WINDOWS_VENV%\Scripts\python.exe" -m cav_recovery.cli %*
exit /b %errorlevel%
