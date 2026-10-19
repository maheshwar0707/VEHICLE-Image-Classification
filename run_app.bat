@echo off
title Fine-Grained Vehicle Classifier (Deep Learning)
echo ========================================================
echo   Launching Fine-Grained Vehicle Classifier ML Web App
echo ========================================================
echo.
py -m streamlit run app.py --server.port 8501
pause
