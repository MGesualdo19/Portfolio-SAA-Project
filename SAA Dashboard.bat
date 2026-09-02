@echo off
REM Launches the SAA desktop app with pythonw.exe so no console window appears.
REM Double-click this file, or use scripts\create_shortcut.py to put it on the Desktop.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m desktop.main
) else (
  start "" pythonw -m desktop.main
)
endlocal
