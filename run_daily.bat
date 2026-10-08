@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo ===== %date% %time% START ===== >> data\scheduler.log
venv\Scripts\python.exe trading\execute.py --live >> data\scheduler.log 2>&1
echo ===== %date% %time% END (exit %errorlevel%) ===== >> data\scheduler.log