@echo off
title TrafficVision AI - Tactical Highway Surveillance System
color 0B
echo ========================================================
echo    TRAFFICVISION AI - SMART HIGHWAY SURVEILLANCE
echo    Launching Real-Time Command Center UI...
echo ========================================================
echo.
cd /d "%~dp0"
echo Starting local web server on port 8080...
start "" http://localhost:8080/dashboard.html
python -m http.server 8080
pause
