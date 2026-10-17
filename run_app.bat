@echo off
title AI Vehicle Classifier
echo ========================================================
echo   Launching AI Fine-Grained Vehicle Classifier Web App
echo ========================================================
echo.
py -m streamlit run app.py --server.port 8501
pause
