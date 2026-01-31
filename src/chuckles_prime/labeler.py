"""Human-in-the-loop labeling web app for parody preferences.

Provides a Flask web app for quickly labeling parody title pairs
(Parody 1 / Parody 2 / Both Bad) and exporting to DPO datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


def load_labels(labels_path: Path) -> dict[str, Any]:
    """Load labels from a JSON file.

    Args:
        labels_path: Path to the labels JSON file.

    Returns:
        Labels dict with 'version' and 'labels' keys.
    """
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "labels": []}


def save_labels(labels_path: Path, data: dict[str, Any]) -> None:
    """Save labels to a JSON file.

    Args:
        labels_path: Path to write the labels JSON file.
        data: Labels dict with 'version' and 'labels' keys.
    """
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_dpo_from_labels(labels_path: Path) -> list[dict[str, Any]]:
    """Convert human labels to DPO preference rows.

    For each label where winner != 'both_bad', produces a DPO row with
    chosen/rejected messages matching the existing dataset schema.

    Args:
        labels_path: Path to labels.json.

    Returns:
        List of dicts with prompt/chosen/rejected keys.
    """
    from chuckles_prime.dataset import DATASET_SYSTEM_PROMPT

    data = load_labels(labels_path)
    rows: list[dict[str, Any]] = []

    for label in data.get("labels", []):
        winner = label.get("winner")
        if winner == "both_bad":
            continue

        if winner == "parody1":
            chosen_text = label["parody1"]
            rejected_text = label["parody2"]
        elif winner == "parody2":
            chosen_text = label["parody2"]
            rejected_text = label["parody1"]
        else:
            continue

        rows.append({
            "prompt": [
                {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Create a phonetically-sound parody of: '{label['input_title']}'",
                },
            ],
            "chosen": [
                {"role": "assistant", "content": chosen_text},
            ],
            "rejected": [
                {"role": "assistant", "content": rejected_text},
            ],
        })

    return rows


def _prepare_items(records: list) -> list[dict[str, Any]]:
    """Extract lightweight card data from GenerationRecords.

    Filters to records with 2+ candidates and no error.

    Args:
        records: List of GenerationRecord objects.

    Returns:
        List of dicts with input_title, parody1, parody2, and scores.
    """
    items = []
    for r in records:
        if r.error is not None or len(r.candidates) < 2:
            continue
        items.append({
            "input_title": r.input_title,
            "parody1": r.candidates[0].text,
            "parody2": r.candidates[1].text,
            "score1": r.candidates[0].phonetic_scores,
            "score2": r.candidates[1].phonetic_scores,
        })
    return items


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chuckles Labeler</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  header { position: sticky; top: 0; z-index: 100; background: #16213e;
           padding: 12px 20px; box-shadow: 0 2px 8px rgba(0,0,0,.4); }
  .stats { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
  .stats span { font-size: 14px; }
  .progress-bar { flex: 1; min-width: 200px; height: 8px; background: #2a2a4a;
                  border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: #0f3460; transition: width .3s; }
  .filters { display: flex; gap: 8px; margin-top: 8px; }
  .filters button { padding: 4px 12px; border: 1px solid #444; background: transparent;
                    color: #ccc; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .filters button.active { background: #0f3460; border-color: #0f3460; color: #fff; }
  .container { max-width: 900px; margin: 20px auto; padding: 0 16px; }
  .card { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 12px;
          border-left: 4px solid transparent; transition: opacity .2s, border-color .2s;
          outline: none; }
  .card.focused { box-shadow: 0 0 0 2px #e94560; }
  .card.labeled { opacity: .55; }
  .card.labeled.w-parody1 { border-left-color: #27ae60; }
  .card.labeled.w-parody2 { border-left-color: #2980b9; }
  .card.labeled.w-both_bad { border-left-color: #c0392b; }
  .card-title { font-size: 15px; color: #aaa; margin-bottom: 10px; }
  .card-title strong { color: #fff; }
  .parodies { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
  .parody { flex: 1; text-align: center; font-size: 18px; font-weight: 600; }
  .parody.p1 { color: #2ecc71; }
  .parody.p2 { color: #3498db; }
  .vs { color: #666; font-size: 13px; }
  .btns { display: flex; gap: 8px; justify-content: center; }
  .btns button { padding: 6px 20px; border: none; border-radius: 4px;
                 font-size: 14px; font-weight: 600; cursor: pointer; color: #fff; }
  .btn1 { background: #27ae60; }
  .btn1:hover { background: #2ecc71; }
  .btn2 { background: #2980b9; }
  .btn2:hover { background: #3498db; }
  .btnx { background: #c0392b; }
  .btnx:hover { background: #e74c3c; }
  .btns button.chosen { outline: 3px solid #fff; }
  .kbd { display: inline-block; background: #333; border-radius: 3px;
         padding: 1px 5px; font-size: 11px; color: #999; margin-left: 4px; }
  .hidden { display: none !important; }
  .help { position: fixed; bottom: 12px; right: 16px; font-size: 12px; color: #555; }
</style>
</head>
<body>
<header>
  <div class="stats">
    <span id="st-count">0 / 0 labeled</span>
    <div class="progress-bar"><div class="progress-fill" id="st-bar" style="width:0%"></div></div>
    <span id="st-breakdown"></span>
  </div>
  <div class="filters">
    <button class="active" data-filter="all">All</button>
    <button data-filter="unlabeled">Unlabeled</button>
    <button data-filter="labeled">Labeled</button>
  </div>
</header>
<div class="container" id="cards"></div>
<div class="help">
  <b>Keys:</b> <span class="kbd">1</span> parody 1 &nbsp;
  <span class="kbd">2</span> parody 2 &nbsp;
  <span class="kbd">x</span> both bad &nbsp;
  <span class="kbd">j/↓</span> next &nbsp;
  <span class="kbd">k/↑</span> prev
</div>
<script>
const ITEMS = __ITEMS_JSON__;
const labels = {};  // input_title -> winner
let focusIdx = 0;
let filter = 'all';

// Seed from server-provided existing labels
const EXISTING = __LABELS_JSON__;
EXISTING.forEach(l => { labels[l.input_title] = l.winner; });

function render() {
  const c = document.getElementById('cards');
  c.innerHTML = '';
  ITEMS.forEach((item, i) => {
    const winner = labels[item.input_title] || null;
    const isLabeled = !!winner;
    if (filter === 'unlabeled' && isLabeled) return;
    if (filter === 'labeled' && !isLabeled) return;
    const div = document.createElement('div');
    div.className = 'card' + (isLabeled ? ' labeled w-' + winner : '');
    div.dataset.idx = i;
    div.tabIndex = -1;
    div.innerHTML = `
      <div class="card-title"><strong>${esc(item.input_title)}</strong></div>
      <div class="parodies">
        <div class="parody p1">${esc(item.parody1)}</div>
        <div class="vs">vs</div>
        <div class="parody p2">${esc(item.parody2)}</div>
      </div>
      <div class="btns">
        <button class="btn1${winner==='parody1'?' chosen':''}" onclick="doLabel(${i},'parody1')">1<span class="kbd">1</span></button>
        <button class="btn2${winner==='parody2'?' chosen':''}" onclick="doLabel(${i},'parody2')">2<span class="kbd">2</span></button>
        <button class="btnx${winner==='both_bad'?' chosen':''}" onclick="doLabel(${i},'both_bad')">Both Bad<span class="kbd">x</span></button>
      </div>`;
    c.appendChild(div);
  });
  updateStats();
  focusCard();
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function updateStats() {
  const total = ITEMS.length;
  const labeled = Object.keys(labels).length;
  const p1 = Object.values(labels).filter(v => v === 'parody1').length;
  const p2 = Object.values(labels).filter(v => v === 'parody2').length;
  const bb = Object.values(labels).filter(v => v === 'both_bad').length;
  document.getElementById('st-count').textContent = `${labeled} / ${total} labeled`;
  document.getElementById('st-bar').style.width = total ? (labeled/total*100)+'%' : '0%';
  document.getElementById('st-breakdown').textContent = `P1: ${p1}  P2: ${p2}  Bad: ${bb}`;
}

function doLabel(idx, winner) {
  const item = ITEMS[idx];
  labels[item.input_title] = winner;
  fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      input_title: item.input_title,
      parody1: item.parody1,
      parody2: item.parody2,
      winner: winner
    })
  });
  render();
  // auto-advance
  const cards = document.querySelectorAll('.card');
  const curCardIdx = Array.from(cards).findIndex(c => parseInt(c.dataset.idx) === idx);
  if (curCardIdx >= 0 && curCardIdx < cards.length - 1) {
    focusIdx = parseInt(cards[curCardIdx + 1].dataset.idx);
  }
  focusCard();
}

function focusCard() {
  document.querySelectorAll('.card.focused').forEach(c => c.classList.remove('focused'));
  const cards = document.querySelectorAll('.card');
  let target = Array.from(cards).find(c => parseInt(c.dataset.idx) === focusIdx);
  if (!target && cards.length) target = cards[0];
  if (target) {
    target.classList.add('focused');
    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    focusIdx = parseInt(target.dataset.idx);
  }
}

function moveFocus(delta) {
  const cards = Array.from(document.querySelectorAll('.card'));
  if (!cards.length) return;
  const curIdx = cards.findIndex(c => parseInt(c.dataset.idx) === focusIdx);
  const next = Math.max(0, Math.min(cards.length - 1, curIdx + delta));
  focusIdx = parseInt(cards[next].dataset.idx);
  focusCard();
}

document.addEventListener('keydown', e => {
  if (e.key === '1') doLabel(focusIdx, 'parody1');
  else if (e.key === '2') doLabel(focusIdx, 'parody2');
  else if (e.key === 'x' || e.key === 'X') doLabel(focusIdx, 'both_bad');
  else if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); moveFocus(1); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); moveFocus(-1); }
});

document.querySelectorAll('.filters button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filter = btn.dataset.filter;
    render();
  });
});

render();
</script>
</body>
</html>"""


def create_app(items: list[dict[str, Any]], labels_path: Path) -> Flask:
    """Create and configure the Flask labeler app.

    Args:
        items: List of card data dicts from _prepare_items.
        labels_path: Path to read/write labels.json.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        existing = load_labels(labels_path)
        items_json = json.dumps(items)
        labels_json = json.dumps(existing.get("labels", []))
        html = _HTML_TEMPLATE.replace("__ITEMS_JSON__", items_json).replace(
            "__LABELS_JSON__", labels_json
        )
        return html

    @app.route("/label", methods=["POST"])
    def label():
        body = request.get_json(force=True)
        input_title = body["input_title"]
        parody1 = body["parody1"]
        parody2 = body["parody2"]
        winner = body["winner"]

        if winner not in ("parody1", "parody2", "both_bad"):
            return jsonify({"error": "invalid winner"}), 400

        data = load_labels(labels_path)
        # Upsert by input_title
        existing_idx = None
        for i, lbl in enumerate(data["labels"]):
            if lbl["input_title"] == input_title:
                existing_idx = i
                break

        entry = {
            "input_title": input_title,
            "parody1": parody1,
            "parody2": parody2,
            "winner": winner,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if existing_idx is not None:
            data["labels"][existing_idx] = entry
        else:
            data["labels"].append(entry)

        save_labels(labels_path, data)

        total = len(items)
        labeled = len(data["labels"])
        return jsonify({"ok": True, "labeled": labeled, "total": total})

    @app.route("/stats")
    def stats():
        data = load_labels(labels_path)
        labeled = len(data.get("labels", []))
        total = len(items)
        winners = [l.get("winner") for l in data.get("labels", [])]
        return jsonify({
            "labeled": labeled,
            "total": total,
            "parody1": winners.count("parody1"),
            "parody2": winners.count("parody2"),
            "both_bad": winners.count("both_bad"),
        })

    return app
