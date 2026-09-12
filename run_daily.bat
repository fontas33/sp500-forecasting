@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python trading/execute.py --live >> data\scheduler.log 2>&1
