#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f "run.bat" ]; then
    echo "[ERROR] run.bat was not found next to this launcher."
    echo "Expected: $(pwd)/run.bat"
    echo
    read -rp "Press Enter to continue..."
    exit 1
fi

case "${1:-}" in
    help|-h|--help)
        bash run.bat help
        exit $?
        ;;
esac

bash run.bat gui
