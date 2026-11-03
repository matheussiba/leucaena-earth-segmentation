@echo off
REM Wrapper: sempre executa o .ps1 (evita o Windows abrir o arquivo no editor).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
