@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "run.bat" (
    echo [ERROR] run.bat was not found next to this launcher.
    echo Expected: "%~dp0run.bat"
    echo.
    pause
    exit /b 1
)

if /i "%~1"=="help" (
    call "%~dp0run.bat" help
    exit /b %ERRORLEVEL%
)
if /i "%~1"=="-h" (
    call "%~dp0run.bat" help
    exit /b %ERRORLEVEL%
)
if /i "%~1"=="--help" (
    call "%~dp0run.bat" help
    exit /b %ERRORLEVEL%
)

call "%~dp0run.bat" gui
exit /b %ERRORLEVEL%
