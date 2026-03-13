@echo off
REM Install all dependencies for test_light_vlms
REM Run from project root: test_light_vlms\install_deps.bat

cd /d "%~dp0.."
echo Installing all dependencies for test_light_vlms...
pip install -r test_light_vlms/requirements_full.txt
echo.
echo Done. Run: python test_light_vlms/main_test.py
