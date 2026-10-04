@echo off
rem Start TradeSense in the background (see start-local.ps1 for options).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-local.ps1" %*
