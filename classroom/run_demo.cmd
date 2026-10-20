@echo off
setlocal
cd /d "%~dp0.."
set "DEMO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%DEMO_PYTHON%" (
    echo Project Python environment was not found.
    echo Create the Python 3.12 .venv and install requirements.txt first.
    echo Instructions: README.md in the project folder.
    pause
    exit /b 1
)
"%DEMO_PYTHON%" "%~dp0demo.py" --trials 30 --rounds 1 --open-figure
set "DEMO_EXIT=%ERRORLEVEL%"
echo.
if not "%DEMO_EXIT%"=="0" echo Demo stopped with an error. Read the message above.
pause
exit /b %DEMO_EXIT%
