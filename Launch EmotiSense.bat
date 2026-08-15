@echo off
setlocal

rem Always run from the folder this file lives in, regardless of where a
rem shortcut to it is placed (e.g. your Desktop) or where it's launched
rem from - %~dp0 is this .bat file's own folder, not the current directory.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Could not find venv\Scripts\python.exe
    echo.
    echo Make sure this file is sitting directly inside the emotion-project
    echo folder itself, next to app\, src\, and venv\ - not moved somewhere
    echo else. If you renamed or recreated your virtual environment, update
    echo the "venv" name below to match.
    echo.
    pause
    exit /b 1
)

if not exist "results\emotion_model.pkl" (
    echo Warning: results\emotion_model.pkl not found - EmotiSense will show
    echo an error until the models are trained. See README.md "Training
    echo Models" if this is a fresh setup.
    echo.
)

title EmotiSense
echo Starting EmotiSense...
echo.
echo Your browser should open automatically in a few seconds.
echo Leave THIS WINDOW OPEN while using the app - closing it (or pressing
echo Ctrl+C) stops the app.
echo.

"venv\Scripts\python.exe" -m streamlit run "app\app.py"

echo.
echo EmotiSense has stopped.
pause
