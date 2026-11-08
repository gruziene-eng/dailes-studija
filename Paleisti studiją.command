#!/bin/zsh
# Dailės studija – paleidimas dukart spustelėjus
cd "$(dirname "$0")"
VENV="$HOME/.venvs/dailes-studija"
UV="$HOME/.local/bin/uv"

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; sys.exit(sys.version_info < (3,10))' 2>/dev/null; then
  echo "Pirmas paleidimas: diegiamas Python 3.12 ir reikalingos bibliotekos (1–3 min)…"
  if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh || { echo "Nepavyko įdiegti uv."; read; exit 1; }
  fi
  rm -rf "$VENV"
  "$UV" venv -q --python 3.12 "$VENV" || { echo "Nepavyko sukurti Python aplinkos."; read; exit 1; }
  VIRTUAL_ENV="$VENV" "$UV" pip install -q -r requirements.txt || { echo "Nepavyko įdiegti bibliotekų."; read; exit 1; }
fi
exec "$VENV/bin/python" -W ignore -m studija.server "$@"
