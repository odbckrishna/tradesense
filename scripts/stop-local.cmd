@echo off
rem Stop the background TradeSense server (see stop-local.ps1 for options).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-local.ps1" %*
