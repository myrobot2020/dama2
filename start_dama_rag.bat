@echo off
setlocal

REM Starts the local Dama RAG web app on http://127.0.0.1:8000
cd /d "%~dp0"

REM Make sure Ollama is running (start it if not already).
start "" /min cmd /c "ollama serve"
timeout /t 2 /nobreak > nul

REM Start the server in a separate window so this script can continue.
start "Dama RAG Server" cmd /k ""C:\Users\ADMIN\miniconda3\python.exe" -m uvicorn local_app:app --host 127.0.0.1 --port 8000"

REM Give the server a moment to start, then open the app in the default browser.
ping 127.0.0.1 -n 3 > nul
start "" "http://127.0.0.1:8000"

endlocal
