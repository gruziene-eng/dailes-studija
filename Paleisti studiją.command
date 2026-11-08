#!/bin/zsh
# Dailės studija – paleidimas dukart spustelėjus
cd "$(dirname "$0")"
VENV="$HOME/.venvs/dailes-studija"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Pirmas paleidimas: diegiamos reikalingos bibliotekos (1–3 min)…"
  python3 -m venv "$VENV" || { echo "Nepavyko sukurti Python aplinkos. Įdiekite Xcode Command Line Tools: xcode-select --install"; read; exit 1; }
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt || { echo "Nepavyko įdiegti bibliotekų."; read; exit 1; }
fi
exec "$VENV/bin/python" -W ignore -m studija.server "$@"
