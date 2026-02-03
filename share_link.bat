@echo off
echo Starting AI Sentiment App...
start "AI Flask Server" cmd /k "python app.py"

echo.
echo Waiting for server to start...
timeout /t 5 >nul

echo.
echo ========================================================
echo Creating Public Link... (Press Ctrl+C to stop)
echo Copy the link below (ending with .serveo.net) and send to your friend!
echo ========================================================
echo.
ssh -R 80:localhost:5000 serveo.net
pause
