#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Poker RL — setup automatico ==="

# --- Python venv (opzionale: commenta se non serve) ---
if [ ! -d "venv" ]; then
    echo "[1/4] creazione virtual environment..."
    python3 -m venv venv
fi

echo "[2/4] attivazione venv..."
source venv/bin/activate

echo "[3/4] installazione dipendenze pip..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e ./pokerl -q

echo "[4/4] verifica installazione..."
python3 -c "
from es_poker import SmallNN, ESAgent, RandomAgent, EvolutionStrategies, TrainingProtocol
from pokerl.game import Game
print('  pokerl  → OK')
print('  es_poker → OK')
"

echo ""
echo "=== Setup completato! ==="
echo "Attiva il venv con:  source venv/bin/activate"
echo "Lancia training:     python -m es_poker.main train --parallel"
echo "Esegui test:          pytest es_poker/tests/"
