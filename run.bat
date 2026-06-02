@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /i "%~1"=="help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help

set "LAUNCH_MODE=console"
if /i "%~1"=="gui" set "LAUNCH_MODE=gui"
set "VENV_RECREATE_DONE=0"
set "IMPORT_REPAIR_DONE=0"

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

:create_venv
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
set "PYW=%CD%\.venv\Scripts\pythonw.exe"
set "DEPS_MARKER=%CD%\.venv\.photo_ai_sorter_deps_ok"

"%PY%" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    if "%VENV_RECREATE_DONE%"=="1" (
        echo [ОШИБКА] Нужен Python 3.10+. Проверьте системный Python и запустите стартер снова.
        goto :fail
    )
    echo [1/3] .venv поврежден или создан старой версией Python — пересоздаю ...
    call :remove_venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось удалить старый .venv. Закройте процессы Python и повторите запуск.
        goto :fail
    )
    set "VENV_RECREATE_DONE=1"
    goto :create_venv
)

echo [2/3] Проверка зависимостей ...
set "NEED_INSTALL=0"
if not exist "%DEPS_MARKER%" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="0" (
    "%PY%" -c "from pathlib import Path; import sys; req=Path('requirements.txt'); marker=Path(r'.venv/.photo_ai_sorter_deps_ok'); sys.exit(0 if marker.exists() and marker.stat().st_mtime >= req.stat().st_mtime else 1)" >nul 2>&1
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo       Установка/обновление pip и packages ...
    "%PY%" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check
    if errorlevel 1 goto :pip_fail
    "%PY%" -m pip install -r requirements.txt --upgrade -q --disable-pip-version-check
    if errorlevel 1 goto :pip_fail
    "%PY%" -c "from pathlib import Path; Path(r'.venv/.photo_ai_sorter_deps_ok').write_text('ok', encoding='utf-8')"
    if errorlevel 1 goto :pip_fail
    echo       Готово.
) else (
    echo       Зависимости уже актуальны.
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    "%PY%" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [2/3] NVIDIA GPU найден — ставлю PyTorch с CUDA ^(cu126^) ...
        rem --force-reinstall --no-deps: a CPU torch of the same version already
        rem satisfies torch>=..., so --upgrade is a no-op; force the CUDA wheel,
        rem --no-deps keeps numpy/pillow (absent from the pytorch index) intact.
        "%PY%" -m pip install -r requirements-gpu.txt --force-reinstall --no-deps -q --disable-pip-version-check
        if errorlevel 1 (
            echo [ПРЕДУПРЕЖДЕНИЕ] Не удалось установить torch+cuda. CLIP будет на CPU.
        ) else (
            echo       PyTorch CUDA установлен.
        )
    )
)

if /i "%~1"=="test" (
    "%PY%" -m pip install -r requirements-dev.txt --upgrade -q --disable-pip-version-check
    if errorlevel 1 goto :pip_fail_dev
)

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
    if "%IMPORT_REPAIR_DONE%"=="0" (
        echo [3/3] Импорты не прошли — пересоздаю .venv начисто и повторяю установку ...
        call :remove_venv
        if errorlevel 1 (
            echo [ОШИБКА] Не удалось удалить старый .venv. Закройте процессы Python и повторите запуск.
            goto :fail
        )
        set "IMPORT_REPAIR_DONE=1"
        set "VENV_RECREATE_DONE=1"
        goto :create_venv
    )
    echo [ОШИБКА] Не удалось импортировать приложение ^(см. сообщение выше^).
    echo Попробуйте вручную: "%PY%" -m pip install -r requirements.txt -v
    goto :fail
)

if /i "%~1"=="test" goto :run_tests

echo.
echo       Запуск GUI ...
echo.
if /i "%LAUNCH_MODE%"=="gui" if exist "%PYW%" (
    start "Photo AI Sorter" "%PYW%" "%CD%\main.py"
    exit /b 0
)

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

:pip_fail_dev
echo.
echo [ОШИБКА] pip install dev-зависимостей завершился с ошибкой.
echo Диагностика: повтор без -q, вывод ниже.
echo.
"%PY%" -m pip install -r requirements-dev.txt --upgrade --disable-pip-version-check
goto :fail

:help
echo Использование:
echo   run.bat           — создать/обновить .venv, зависимости, проверки, запуск GUI
echo   run.bat gui       — то же, но GUI запускается через pythonw.exe для двойного клика
echo   run.bat test      — то же, затем pytest -q
echo   run.bat help      — эта справка
echo.
exit /b 0

:remove_venv
set "REMOVE_TRIES=0"
:remove_venv_retry
if not exist ".venv" exit /b 0
rmdir /s /q ".venv" >nul 2>&1
if not exist ".venv" exit /b 0
set /a REMOVE_TRIES+=1 >nul
if %REMOVE_TRIES% LSS 5 (
    timeout /t 2 /nobreak >nul
    goto :remove_venv_retry
)
exit /b 1

:fail
echo.
pause
exit /b 1
