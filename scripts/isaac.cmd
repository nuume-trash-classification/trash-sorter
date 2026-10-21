@echo off
rem Run a script with the Isaac Sim Python.  Usage:  scripts\isaac.cmd sim\run_sorter.py --perception gt
rem Set ISAACSIM_PYTHON to your Isaac Sim venv python if it is not at ..\env_isaacsim (next to this repo).
setlocal
if "%ISAACSIM_PYTHON%"=="" set "ISAACSIM_PYTHON=%~dp0..\..\env_isaacsim\Scripts\python.exe"
if not exist "%ISAACSIM_PYTHON%" (
  echo Isaac Sim Python not found: %ISAACSIM_PYTHON%
  echo Set ISAACSIM_PYTHON, e.g.  set ISAACSIM_PYTHON=D:\isaac\env_isaacsim\Scripts\python.exe
  exit /b 1
)
set OMNI_KIT_ACCEPT_EULA=YES
set PYTHONUTF8=1
"%ISAACSIM_PYTHON%" %*
