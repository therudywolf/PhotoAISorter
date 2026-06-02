@echo off
rem ===========================================================================
rem  Photo AI Sorter - single launcher for Windows.
rem  Double-click this file. On first run it sets up .venv + dependencies,
rem  drops a "Photo AI Sorter" shortcut on the Desktop, then starts the GUI.
rem  Engine: run.bat (run.bat help / test for console use).
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul 2>&1

if /i "%~1"=="help"   ( call "%~dp0run.bat" help & exit /b %ERRORLEVEL% )
if /i "%~1"=="-h"     ( call "%~dp0run.bat" help & exit /b %ERRORLEVEL% )
if /i "%~1"=="--help" ( call "%~dp0run.bat" help & exit /b %ERRORLEVEL% )

if not exist "%~dp0run.bat" (
    echo [ОШИБКА] Рядом с этим файлом не найден run.bat.
    echo Ожидается: "%~dp0run.bat"
    echo.
    pause
    exit /b 1
)

rem One-time: create a Desktop shortcut so next launch is one click away.
if not exist "%~dp0.desktop_shortcut_done" call :make_shortcut

call "%~dp0run.bat" gui
exit /b %ERRORLEVEL%

:make_shortcut
set "PAS_TARGET=%~f0"
set "PAS_WORKDIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $s = New-Object -ComObject WScript.Shell; $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Photo AI Sorter.lnk'; $sc = $s.CreateShortcut($lnk); $sc.TargetPath = $env:PAS_TARGET; $sc.WorkingDirectory = $env:PAS_WORKDIR; $sc.IconLocation = $env:SystemRoot + '\System32\imageres.dll,109'; $sc.Description = 'Photo AI Sorter'; $sc.Save() } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [i] Ярлык на рабочем столе создать не удалось ^(не критично, запуск продолжится^).
) else (
    echo [i] Ярлык «Photo AI Sorter» создан на рабочем столе.
)
rem Mark done either way so we do not retry on every launch.
> "%~dp0.desktop_shortcut_done" echo done
exit /b 0
