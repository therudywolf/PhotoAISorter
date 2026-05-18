#!/usr/bin/env bash
# Wrapper for documentation compatibility — canonical launcher is run.sh
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" "$@"
