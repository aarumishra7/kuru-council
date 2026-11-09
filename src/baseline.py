"""Step 1: single-agent baseline on the crises.
  full   - one advisor holding ALL reports (can the model find the best option at all?)
  public - the king alone with only the public headlines (does the trap work?)
Output: baseline.jsonl and an accuracy table.
"""
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import sys
from common import MODELS, BIG_MODEL, ask, letter

K = 3
SYSTEM = ("You advise the King of Hastinapura. Choose the option that truly best serves the kingdom. "
          "Think briefly, then end with: DECISION: <A/B/C>")
scenarios = json.load(open("data/scenarios.json"))


def prompt(s, info):
    p = f"{s['crisis']}\nWhat matters: {s['measure']}.\n\nPublic facts:\n{s['public']}"
    if info == "full":
        p += "\n\nPrivate reports:\n" + "\n".join(f"- {r}" for r in s["reports"])
    return p


def run(job):
    s, m, info, k = job
    a = letter(ask(m, SYSTEM, prompt(s, info), max_tokens=700), "DECISION")
    return {"sid": s["sid"], "model": m, "info": info, "sample": k, "answer": a,
            "correct": a == s["correct"], "chose_trap": a == s["target"]}


big = "big" in sys.argv          # python src/baseline.py big  -> larger model only, saved to baseline_big.jsonl
models = [BIG_MODEL] if big else MODELS
jobs = [(s, m, info, k) for s in scenarios for m in models for info in ("full", "public") for k in range(K)]
with ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(run, jobs))
with open("data/baseline_big.jsonl" if big else "data/baseline.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)

acc = defaultdict(list)
for r in rows:
    acc[(r["model"], r["info"])].append(r)
print(f"{'model':40} {'info':7} {'best':>6} {'trap':>6} {'unparsed':>9}")
for (m, info), rs in sorted(acc.items()):
    n = len(rs)
    print(f"{m:40} {info:7} {sum(r['correct'] for r in rs)/n:6.0%} {sum(r['chose_trap'] for r in rs)/n:6.0%} "
          f"{sum(r['answer'] is None for r in rs)/n:9.0%}")
