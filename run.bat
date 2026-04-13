@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /i "%~1"=="help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help

rem UTF-8 в консоли и в Python без лишних сюрпризов с путями/логами
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem Локальный ffmpeg (essentials) рядом с проектом — без установки в систему
if exist "%~dp0ffmpeg-runtime\bin\ffmpeg.exe" (
    set "PATH=%~dp0ffmpeg-runtime\bin;%PATH%"
)

title Photo AI Sorter

echo.
echo  ========================================
echo   Photo AI Sorter — запуск
echo  ========================================
echo.

if not exist "requirements.txt" (
    echo [ОШИБКА] Не найден requirements.txt в папке проекта.
    echo Ожидается: "%~dp0requirements.txt"
    goto :fail
)

set "HAVE_PY=0"
where python >nul 2>&1 && set "HAVE_PY=1"
if "%HAVE_PY%"=="0" (
    where py >nul 2>&1 && set "HAVE_PY=2"
)
if "%HAVE_PY%"=="0" (
    echo [ОШИБКА] Не найден Python. Нужен Python 3.10+ в PATH или лаунчер `py`.
    echo Скачать: https://www.python.org/downloads/ ^(отметьте «Add python.exe to PATH»^).
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Создание виртуального окружения .venv ...
    if "%HAVE_PY%"=="1" (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать venv.
        goto :fail
    )
    echo       Готово.
) else (
    echo [1/3] Виртуальное окружение .venv уже есть.
)

set "PY=%CD%\.venv\Scripts\python.exe"

echo [2/3] Обновление pip и установка/обновление зависимостей ...
"%PY%" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check
if errorlevel 1 goto :pip_fail
"%PY%" -m pip install -r requirements.txt --upgrade -q --disable-pip-version-check
if errorlevel 1 goto :pip_fail
echo       Готово.

echo [3/3] Проверка окружения ...
"%PY%" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
    echo [ОШИБКА] Нужен Python 3.10+. В .venv сейчас другая версия — удалите папку .venv и запустите скрипт снова.
    goto :fail
)

"%PY%" -m compileall -q app main.py
if errorlevel 1 (
    echo [ОШИБКА] Ошибка компиляции Python ^(синтаксис^).
    goto :fail
)
"%PY%" -c "import app.gui; import app.gui_duplicates; import app.duplicate_finder; import app.duplicate_worker; import app.signature_db; import app.worker; import app.lm_studio; import app.settings_store; import app.video_frames; print('Проверка импорта: OK')"
if errorlevel 1 (
    echo [ОШИБКА] Не удалось импортировать приложение ^(см. сообщение выше^).
    echo Попробуйте вручную: "%PY%" -m pip install -r requirements.txt -v
    goto :fail
)

if /i "%~1"=="test" goto :run_tests

echo.
echo       Запуск GUI ...
echo.
"%PY%" main.py
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ВЫХОД] Код ошибки: %EXITCODE%
    goto :fail
)
echo.
echo Приложение закрыто нормально.
exit /b 0

:run_tests
echo.
echo       Запуск тестов ^(pytest^) ...
echo.
"%PY%" -m pytest -q
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [ТЕСТЫ] Код выхода: %EXITCODE%
    goto :fail
)
echo.
echo Тесты прошли успешно.
exit /b 0

:pip_fail
echo.
echo [ОШИБКА] pip install завершился с ошибкой.
echo Диагностика: повтор без -q, вывод ниже.
echo.
"%PY%" -m pip install -r requirements.txt --upgrade --disable-pip-version-check
goto :fail

:help
echo Использование:
echo   run.bat           — создать/обновить .venv, зависимости, проверки, запуск GUI
echo   run.bat test      — то же, затем pytest -q
echo   run.bat help      — эта справка
echo.
exit /b 0

:fail
echo.
pause
exit /b 1
