@echo off
echo Starting PDF Figure Extractor Dashboard...
echo.
echo The dashboard will open in your default web browser.
echo Press Ctrl+C to stop the server when done.
echo.
py -3.10 -m streamlit run pdf_extractor_gui.py
if errorlevel 1 (
    echo.
    echo Error occurred! Make sure you ran install.bat first.
    echo.
    pause
)