"""
fusion.py — Research sandbox for L1/L2 score fusion.

Current production design (freehiero):
  L1 (regex) → certain → return
            → uncertain → L2 (Qwen) → return

Research goal:
  Both L1 and L2 contribute a score; a formula combines them.
  Even when L1 is "certain", L2 can override.

  final_score = α * l1_score(text) + (1 - α) * l2_prob(text)
  prediction  = final_score > threshold

Experiments in this file:
  1. l1_score()   — turn regex patterns into a float score
  2. l2_prob()    — parse Qwen's response into a probability
  3. fuse()       — combine with tunable α and threshold
  4. evaluate()   — measure precision / recall on training_data.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from detector import (
    _FREE_WORDS, _FOOD_WORDS,
    _PATTERN_FREE_FOOD, _PATTERN_EVENT_FOOD,
    _PATTERN_HAS_CATERING, _PATTERN_NOT_FOOD,
)

DATA_FILE = Path(__file__).parent.parent / "data" / "training_data.json"


# ── L1: soft score ────────────────────────────────────────────────────────────

def l1_score(text: str) -> float:
    """Return a confidence score in [0, 1] from regex patterns."""
    score = 0.0

    if _PATTERN_NOT_FOOD.search(text):
        return 0.0                          # hard negative override

    if _PATTERN_FREE_FOOD.search(text):    score += 0.80
    if _PATTERN_EVENT_FOOD.search(text):   score += 0.60
    if _PATTERN_HAS_CATERING.search(text): score += 0.50

    # Partial signals (weaker evidence)
    has_free = bool(re.search(_FREE_WORDS, text, re.IGNORECASE))
    has_food = bool(re.search(_FOOD_WORDS, text, re.IGNORECASE))
    if has_free and not _PATTERN_FREE_FOOD.search(text): score += 0.20
    if has_food and not _PATTERN_FREE_FOOD.search(text): score += 0.10

    return min(score, 1.0)


# ── L2: Qwen probability ──────────────────────────────────────────────────────

_PROMPT = (
    "判斷以下貼文是否在提供免費食物或飲料（可以現在就去拿）。"
    "只回答 yes 或 no。\n\n貼文：{text}"
)


def l2_prob(text: str) -> float:
    """Query Qwen via Ollama; return 1.0 for yes, 0.0 for no."""
    if not config.OLLAMA_ENABLED:
        return 0.5                          # neutral when disabled
    try:
        resp = httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": config.OLLAMA_MODEL, "prompt": _PROMPT.format(text=text[:500]), "stream": False},
            timeout=15,
        )
        answer = resp.json().get("response", "").strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0
    except Exception:
        return 0.5


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse(text: str, alpha: float = 0.35, threshold: float = 0.50) -> bool:
    """
    Combine L1 and L2 scores.

    alpha     : weight of L1 score (1-alpha goes to L2)
    threshold : minimum final score to predict True

    Skip L2 when l1_score == 0.0 (no signal at all → definitely negative).
    """
    s1 = l1_score(text)
    if s1 == 0.0:
        return False
    s2 = l2_prob(text)
    return (alpha * s1 + (1 - alpha) * s2) > threshold


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(alpha: float = 0.35, threshold: float = 0.50) -> dict:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    tp = fp = tn = fn = 0
    for d in data:
        pred = fuse(d["text"], alpha=alpha, threshold=threshold)
        true = bool(d["label"])
        if pred and true:   tp += 1
        elif pred and not true: fp += 1
        elif not pred and true: fn += 1
        else:               tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


if __name__ == "__main__":
    # Quick grid search over alpha and threshold
    print(f"{'alpha':>6} {'thresh':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for a in [0.2, 0.35, 0.5]:
        for t in [0.4, 0.5, 0.6]:
            r = evaluate(alpha=a, threshold=t)
            print(f"{a:>6.2f} {t:>6.2f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
