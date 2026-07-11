@echo off
setlocal

if "%~1"=="" (
    echo.
    echo Drag a deck screenshot onto debug-deck.cmd
    echo or run:
    echo debug-deck.cmd "C:\path\to\screenshot.webp"
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0"

set "PVZH_DEBUG=1"
set "INPUT_IMAGE=%~1"
set "OUTPUT_DIR=cards"

echo.
echo ==========================================
echo PvZH DECK DEBUG
echo ==========================================
echo Image: %INPUT_IMAGE%
echo.

echo [1/3] Splitting cards...
node split-deck.js "%INPUT_IMAGE%" "%OUTPUT_DIR%"

if errorlevel 1 (
    echo.
    echo Card splitting failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Running OCR...
node read-deck.js "%INPUT_IMAGE%" "%OUTPUT_DIR%"

if errorlevel 1 (
    echo.
    echo OCR failed.
    pause
    exit /b 1
)

if exist "reference_index" (
    echo.
    echo [3/3] Identifying cards...
    py identify-cards.py "%OUTPUT_DIR%" card_data.json reference_index

    if errorlevel 1 (
        echo.
        echo Card identification failed, but the split and OCR files still exist.
    )
) else (
    echo.
    echo reference_index was not found locally.
    echo Skipping card identification.
    echo The split cards and OCR debug images were still generated.
)

echo.
echo ==========================================
echo COMPLETE
echo ==========================================
echo Output folder:
echo %CD%\%OUTPUT_DIR%
echo.

start "" explorer "%CD%\%OUTPUT_DIR%"

pause