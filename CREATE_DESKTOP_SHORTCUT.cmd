@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "LAUNCHER=%~dp0START_Photo_AI_Sorter.cmd"
set "APPDIR=%~dp0"

if not exist "%LAUNCHER%" (
    echo [ERROR] Launcher was not found:
    echo "%LAUNCHER%"
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$shell = New-Object -ComObject WScript.Shell; " ^
  "$desktop = [Environment]::GetFolderPath('Desktop'); " ^
  "$shortcut = $shell.CreateShortcut((Join-Path $desktop 'Photo AI Sorter.lnk')); " ^
  "$shortcut.TargetPath = $env:LAUNCHER; " ^
  "$shortcut.WorkingDirectory = $env:APPDIR; " ^
  "$shortcut.Description = 'Photo AI Sorter'; " ^
  "$shortcut.IconLocation = $env:SystemRoot + '\System32\imageres.dll,109'; " ^
  "$shortcut.Save()"

if errorlevel 1 (
    echo [ERROR] Failed to create desktop shortcut.
    echo.
    pause
    exit /b 1
)

echo Desktop shortcut created: Photo AI Sorter
echo.
pause
exit /b 0
