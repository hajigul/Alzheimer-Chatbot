@echo off
REM ============================================================
REM  Alzheimer's Support Chatbot - double-click launcher
REM  Put this file in your project root:
REM     D:\23h1710\Alzheimer_chatboot_\
REM ============================================================

REM Move to the folder this .bat file lives in
cd /d "%~dp0"

echo ============================================================
echo   Starting Alzheimer's Support Chatbot...
echo   A browser tab will open at http://localhost:8501
echo   Keep THIS window open while using the chatbot.
echo   Close this window to stop the chatbot.
echo ============================================================
echo.

REM Activate the conda environment
call conda activate alzheimer

REM Launch the Streamlit GUI
streamlit run src\gui.py --server.fileWatcherType none

REM Keep window open if something goes wrong
pause
