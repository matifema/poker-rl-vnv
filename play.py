#!/usr/bin/env python3
"""server per giocare contro l'agente addestrato (browser)"""

import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "pokerl"))

from pokerl.game import Game
from es_poker.agent import ESAgent, masked_argmax
from es_poker.features import extract_features
from es_poker.network import SmallNN

app = Flask(__name__)

GAME_CFG = {"num_players": 2, "start_credits": 1000, "big_blind": 20, "small_blind": 10}
ACTION_NAMES = ["FOLD", "CHECK", "CALL", "RAISE_10", "RAISE_25", "RAISE_50", "ALL_IN"]

game: Game | None = None
agent: ESAgent | None = None


def _state_to_json(state) -> dict:
    vu = state.valid_actions
    return {
        "player": int(state.player),
        "pot": float(state.pot),
        "credits": [float(c) for c in state.credits],
        "pending": [float(p) for p in state.pending_bets],
        "hole": [
            {"rank": int(state.player_cards[0].rank), "suit": int(state.player_cards[0].suit)},
            {"rank": int(state.player_cards[1].rank), "suit": int(state.player_cards[1].suit)},
        ],
        "community": [
            {"rank": int(c.rank), "suit": int(c.suit)} for c in state.community_cards
        ],
        "valid_actions": [i for i, v in enumerate(vu) if v],
        "action_names": [ACTION_NAMES[i] for i, v in enumerate(vu) if v],
        "turn": int(state.turn),
        "hand_over": False,
        "winner": None,
        "payoffs": None,
    }


def _bot_act(state) -> int:
    return agent(state)


@app.route("/api/new", methods=["POST"])
def api_new():
    global game
    game = Game(**GAME_CFG)
    game.reset()
    data = _state_to_json(game.active_state)
    data["big_blind"] = GAME_CFG["big_blind"]
    data["start_credits"] = GAME_CFG["start_credits"]
    return jsonify(data)


@app.route("/api/act", methods=["POST"])
def api_act():
    global game
    data = request.get_json()
    action = int(data["action"])

    state = game.active_state
    assert state.valid_actions[action], f"azione invalida: {action}"

    done, hand_over, _ = game.step(action)
    steps = []

    if not hand_over:
        while not done and game.active_player != 0:
            bot_action = _bot_act(game.active_state)
            game.step(bot_action)

        if game.active_player == 0 or done:
            s = _state_to_json(game.active_state)

        if done:
            s["hand_over"] = True
            s["winner"] = int(np.argmax(game.payoffs))
            s["payoffs"] = [float(p) for p in game.payoffs]
    else:
        s = _state_to_json(game.active_state)
        s["hand_over"] = True
        s["winner"] = int(np.argmax(game.payoffs))
        s["payoffs"] = [float(p) for p in game.payoffs]

    return jsonify(s)


@app.route("/")
def index():
    return app.send_static_file("play.html")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Poker RL — Play vs Agent</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
#app { max-width: 520px; width: 100%; padding: 20px; }
h1 { text-align: center; font-size: 1.5rem; margin-bottom: 16px; color: #e94560; }
.card-row { display: flex; gap: 8px; margin: 8px 0; }
.card { width: 56px; height: 80px; border-radius: 8px; background: #fff; color: #111; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: bold; }
.card.red { color: #c0392b; }
.card.back { background: #2c3e50; color: #2c3e50; }
.suit { font-size: 1.5rem; }
.label { font-size: 0.75rem; color: #888; margin-top: 10px; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 1px; }
.stats { display: flex; gap: 20px; margin: 8px 0; font-size: 0.95rem; }
.stats div { background: #16213e; padding: 8px 14px; border-radius: 6px; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.actions button { flex: 1; min-width: 80px; padding: 12px 8px; border: none; border-radius: 8px; background: #0f3460; color: #eee; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: .15s; }
.actions button:hover { background: #e94560; }
.actions button:disabled { opacity: 0.3; cursor: not-allowed; }
.result { text-align: center; margin-top: 16px; padding: 12px; border-radius: 8px; font-size: 1.1rem; font-weight: bold; }
.result.win { background: #1b5e20; }
.result.lose { background: #b71c1c; }
.result.tie { background: #555; }
#new-btn { margin-top: 12px; width: 100%; padding: 12px; border: none; border-radius: 8px; background: #e94560; color: #fff; font-size: 1rem; font-weight: bold; cursor: pointer; }
#score { text-align: center; font-size: 0.9rem; margin-bottom: 8px; color: #aaa; }
</style>
</head>
<body>
<div id="app">
  <h1>Poker RL — Play vs Agent</h1>
  <div id="score"></div>
  <div class="label">your hand</div>
  <div class="card-row" id="hole"></div>
  <div class="label">community cards</div>
  <div class="card-row" id="community"></div>
  <div class="stats">
    <div>pot <b id="pot">0</b></div>
    <div>stack <b id="stack">1000</b></div>
    <div>blinds <b id="blinds">10/20</b></div>
  </div>
  <div class="actions" id="actions"></div>
  <div id="result"></div>
  <button id="new-btn" style="display:none">new hand</button>
</div>

<script>
const SUITS = ['♠', '♥', '♦', '♣'];
const RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
const RED_SUITS = [1, 2];

let total_profit = 0;
let hands_played = 0;

function render_card(row_id, cards) {
  const el = document.getElementById(row_id);
  el.innerHTML = '';
  cards.forEach(c => {
    const div = document.createElement('div');
    div.className = 'card' + (RED_SUITS.includes(c.suit) ? ' red' : '');
    div.innerHTML = `<span>${RANKS[c.rank]}</span><span class="suit">${SUITS[c.suit]}</span>`;
    el.appendChild(div);
  });
  const rem = 5 - cards.length;
  for (let i = 0; i < rem; i++) {
    const div = document.createElement('div');
    div.className = 'card back';
    div.innerHTML = '<span></span><span class="suit"></span>';
    el.appendChild(div);
  }
}

function render(state) {
  render_card('hole', state.hole);
  render_card('community', state.community);
  document.getElementById('pot').textContent = state.pot;
  document.getElementById('stack').textContent = state.credits[0];
  document.getElementById('blinds').textContent = (state.big_blind||20)/2 + '/' + (state.big_blind||20);

  const actions_el = document.getElementById('actions');
  actions_el.innerHTML = '';
  if (state.hand_over) {
    const res = document.getElementById('result');
    const profit = state.payoffs[0];
    total_profit += profit;
    hands_played++;
    document.getElementById('score').textContent =
      `score: ${total_profit > 0 ? '+' : ''}${total_profit.toFixed(0)} chips over ${hands_played} hands`;

    if (profit > 0) { res.className = 'result win'; res.textContent = `You won +${profit} chips!`; }
    else if (profit < 0) { res.className = 'result lose'; res.textContent = `You lost ${profit} chips`; }
    else { res.className = 'result tie'; res.textContent = 'Tie!'; }
    document.getElementById('new-btn').style.display = 'block';
  } else {
    document.getElementById('result').textContent = '';
    document.getElementById('new-btn').style.display = 'none';
    state.valid_actions.forEach((a, i) => {
      const btn = document.createElement('button');
      btn.textContent = state.action_names[i];
      btn.onclick = () => act(a);
      actions_el.appendChild(btn);
    });
  }
}

async function act(action) {
  document.querySelectorAll('#actions button').forEach(b => b.disabled = true);
  const r = await fetch('/api/act', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action}) });
  render(await r.json());
}

async function new_game() {
  const r = await fetch('/api/new', { method: 'POST' });
  document.getElementById('result').textContent = '';
  document.getElementById('new-btn').style.display = 'none';
  render(await r.json());
}

document.getElementById('new-btn').onclick = new_game;
new_game();
</script>
</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="gioca contro l'agente nel browser")
    parser.add_argument("weights", help="percorso file .npz (es. training_output/agent_A_coev.npz)")
    parser.add_argument("--port", type=int, default=8080, help="porta HTTP")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global agent
    nn = SmallNN()
    data = np.load(args.weights)
    nn.set_weights(data["weights"])
    agent = ESAgent(nn)
    print(f"agente caricato da {args.weights}")

    import os
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(exist_ok=True)
    (static_dir / "play.html").write_text(HTML)

    app._static_folder = str(static_dir)
    print(f"apri http://localhost:{args.port} nel browser")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
