@echo off
REM Batch multi-rede (patch 256 / 30 ep / run-001). Ver scripts/phd-run/run_arch_batch.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\phd-run\run_arch_batch.ps1" %*
exit /b %ERRORLEVEL%
