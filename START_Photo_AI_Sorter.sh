#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv-linux"
LAUNCH_MODE="console"
VENV_RECREATE_DONE=0
IMPORT_REPAIR_DONE=0

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Локальный ffmpeg рядом с проектом
if [ -d "ffmpeg-runtime/bin" ] && [ -x "ffmpeg-runtime/bin/ffmpeg" ]; then
    export PATH="$(pwd)/ffmpeg-runtime/bin:$PATH"
fi

show_help() {
    echo "Использование:"
    echo "  ./START_Photo_AI_Sorter.sh           — создать/обновить .venv, зависимости, проверки, запуск GUI"
    echo "  ./START_Photo_AI_Sorter.sh gui       — то же (GUI)"
    echo "  ./START_Photo_AI_Sorter.sh test      — то же, затем pytest -q"
    echo "  ./START_Photo_AI_Sorter.sh help      — эта справка"
    echo
    exit 0
}

fail() {
    echo
    echo "Нажмите Enter для выхода..."
    read -r
    exit 1
}

remove_venv() {
    local tries=0
    while [ -d "$VENV_DIR" ] && [ $tries -lt 5 ]; do
        rm -rf "$VENV_DIR" 2>/dev/null || true
        if [ -d "$VENV_DIR" ]; then
            tries=$((tries + 1))
            sleep 2
        fi
    done
    [ ! -d "$VENV_DIR" ]
}

case "${1:-}" in
    help|-h|--help) show_help ;;
    gui) LAUNCH_MODE="gui" ;;
    test) LAUNCH_MODE="test" ;;
esac

echo
echo " ========================================"
echo "  Photo AI Sorter — запуск"
echo " ========================================"
echo

if [ ! -f "requirements.txt" ]; then
    echo "[ОШИБКА] Не найден requirements.txt в папке проекта."
    echo "Ожидается: $(pwd)/requirements.txt"
    fail
fi

if ! command -v python3 &>/dev/null; then
    echo "[ОШИБКА] Не найден python3. Нужен Python 3.10+."
    echo "Установите: sudo apt install python3 python3-venv python3-pip"
    fail
fi

create_venv() {
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo "[1/3] Создание виртуального окружения $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "[ОШИБКА] Не удалось создать venv."
            fail
        fi
        echo "      Готово."
    else
        echo "[1/3] Виртуальное окружение $VENV_DIR уже есть."
    fi
}

create_venv

PY="$(pwd)/$VENV_DIR/bin/python"
DEPS_MARKER="$(pwd)/$VENV_DIR/.photo_ai_sorter_deps_ok"

if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
    if [ "$VENV_RECREATE_DONE" -eq 1 ]; then
        echo "[ОШИБКА] Нужен Python 3.10+. Проверьте системный Python и запустите снова."
        fail
    fi
    echo "[1/3] $VENV_DIR повреждён или создан старой версией Python — пересоздаю ..."
    if ! remove_venv; then
        echo "[ОШИБКА] Не удалось удалить старый .venv."
        fail
    fi
    VENV_RECREATE_DONE=1
    create_venv
    PY="$(pwd)/$VENV_DIR/bin/python"
fi

echo "[2/3] Проверка зависимостей ..."
NEED_INSTALL=0
if [ ! -f "$DEPS_MARKER" ]; then
    NEED_INSTALL=1
else
    if ! "$PY" -c "
from pathlib import Path; import sys
req = Path('requirements.txt')
marker = Path('$VENV_DIR/.photo_ai_sorter_deps_ok')
sys.exit(0 if marker.exists() and marker.stat().st_mtime >= req.stat().st_mtime else 1)
" 2>/dev/null; then
        NEED_INSTALL=1
    fi
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "      Установка/обновление pip и packages ..."
    "$PY" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check || {
        echo "[ОШИБКА] pip install завершился с ошибкой."
        "$PY" -m pip install -r requirements.txt --upgrade --disable-pip-version-check
        fail
    }
    "$PY" -m pip install -r requirements.txt --upgrade -q --disable-pip-version-check || {
        echo "[ОШИБКА] pip install завершился с ошибкой."
        "$PY" -m pip install -r requirements.txt --upgrade --disable-pip-version-check
        fail
    }
    "$PY" -c "from pathlib import Path; Path('$VENV_DIR/.photo_ai_sorter_deps_ok').write_text('ok', encoding='utf-8')"
    echo "      Готово."
else
    echo "      Зависимости уже актуальны."
fi

echo "[3/3] Проверка окружения ..."
if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"; then
    echo "[ОШИБКА] Нужен Python 3.10+. Удалите $VENV_DIR и запустите скрипт снова."
    fail
fi

"$PY" -m compileall -q app main.py || {
    echo "[ОШИБКА] Ошибка компиляции Python (синтаксис)."
    fail
}

if ! "$PY" -c "import app.gui; import app.gui_duplicates; import app.duplicate_finder; import app.duplicate_worker; import app.signature_db; import app.worker; import app.lm_studio; import app.settings_store; import app.video_frames; print('Проверка импорта: OK')"; then
    if [ "$IMPORT_REPAIR_DONE" -eq 0 ]; then
        echo "[3/3] Импорты не прошли — пересоздаю $VENV_DIR начисто ..."
        if ! remove_venv; then
            echo "[ОШИБКА] Не удалось удалить старый $VENV_DIR."
            fail
        fi
        IMPORT_REPAIR_DONE=1
        VENV_RECREATE_DONE=1
        create_venv
        PY="$(pwd)/$VENV_DIR/bin/python"
        echo "      Установка зависимостей ..."
        "$PY" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check
        "$PY" -m pip install -r requirements.txt --upgrade -q --disable-pip-version-check
        "$PY" -c "from pathlib import Path; Path('$VENV_DIR/.photo_ai_sorter_deps_ok').write_text('ok', encoding='utf-8')"
        if ! "$PY" -c "import app.gui; import app.gui_duplicates; import app.duplicate_finder; import app.duplicate_worker; import app.signature_db; import app.worker; import app.lm_studio; import app.settings_store; import app.video_frames; print('Проверка импорта: OK')"; then
            echo "[ОШИБКА] Не удалось импортировать приложение."
            echo "Попробуйте вручную: $PY -m pip install -r requirements.txt -v"
            fail
        fi
    else
        echo "[ОШИБКА] Не удалось импортировать приложение."
        echo "Попробуйте вручную: $PY -m pip install -r requirements.txt -v"
        fail
    fi
fi

if [ "$LAUNCH_MODE" = "test" ]; then
    echo
    echo "      Запуск тестов (pytest) ..."
    echo
    "$PY" -m pytest -q
    EXITCODE=$?
    if [ $EXITCODE -ne 0 ]; then
        echo
        echo "[ТЕСТЫ] Код выхода: $EXITCODE"
        fail
    fi
    echo
    echo "Тесты прошли успешно."
    exit 0
fi

echo
echo "      Запуск GUI ..."
echo
"$PY" main.py
EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo
    echo "[ВЫХОД] Код ошибки: $EXITCODE"
    fail
fi
echo
echo "Приложение закрыто нормально."
exit 0
