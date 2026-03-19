@echo off
REM Install MobileVLM for light VLM benchmarks.
REM Run from project root: test\vlms_light\install_mobilevlm.bat

echo Installing MobileVLM...
pip install git+https://github.com/Meituan-AutoML/MobileVLM.git
if %ERRORLEVEL% neq 0 (
    echo.
    echo If pip install fails, clone manually:
    echo   git clone https://github.com/Meituan-AutoML/MobileVLM.git
    echo   set PYTHONPATH=path\to\MobileVLM;%%PYTHONPATH%%
)
echo Done.
