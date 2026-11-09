"""King replay: re-judge every recorded crisis in games.jsonl with a different king, keeping the councils' words fixed.
King variants:
  mistral        - same model as the original king (replay control: same prompt, fresh sample)
  llama, qwen    - the other two small models
  big            - BIG_MODEL (larger model)
  mistral_blind  - Mistral king that does NOT see the council's vote tally
  big_blind      - larger king that does NOT see the vote tally
Output: king_replay.jsonl (resumes if interrupted).
"""
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import MODELS, BIG_MODEL, ask, letter
from council import KING

VARIANTS = {"mistral": (MODELS[1], True), "llama": (MODELS[0], True), "qwen": (MODELS[2], True),
            "big": (BIG_MODEL, True), "mistral_blind": (MODELS[1], False), "big_blind": (BIG_MODEL, False)}
OUT = "data/king_replay.jsonl"
scen = {s["sid"]: s for s in json.load(open("data/scenarios.json"))}
games = [json.loads(l) for l in open("data/games.jsonl")]


def prompts(g):
    """Rebuild, crisis by crisis, exactly what the original king saw."""
    history = []
    for cr in g["crises"]:
        sc = scen[cr["sid"]]
        r = cr["crisis"]
        if g["exile"] and r == g["exile"]["after_crisis"] + 1:
            ex = g["exile"]["exiled"]
            history.append(f"The council exiled {ex}." if ex else "The council failed to exile anyone.")
        brief = f"CRISIS {r}. {sc['crisis']}\nWhat matters: {sc['measure']}.\n\nPublic facts:\n{sc['public']}"
        discussion = "\n".join(f'{m["seat"]}: {m["text"]}' for m in cr["said"])
        tally = ", ".join(f"{k}: {c}" for k, c in sorted(Counter(v for v in cr["votes"].values() if v).items()))
        hist = "\n".join(history)
        yield cr, hist, brief, discussion, tally
        truth = ", ".join(f"{k}={v}" for k, v in sorted(sc["true_values"].items()))
        history.append(f"Crisis {r}: " + "; ".join(f'{m["seat"]} advised {m["advice"]}' for m in cr["said"])
                       + f". Council voted {cr['council']}; the king chose {cr['king']}. Revealed true outcomes: {truth}"
                       f" (best was {sc['correct']}).")


def judge(job):
    g, cr, hist, brief, discussion, tally, variant = job
    model, sees_tally = VARIANTS[variant]
    vote_line = f"Council vote: {tally or 'no valid votes'}.\n\n" if sees_tally else ""
    t = ask(model, KING, hist + f"\n\n{brief}\n\nYour council spoke:\n{discussion}\n\n{vote_line}"
            "Think briefly, then end with: DECISION: <A/B/C>", max_tokens=800)
    return {"game": g["game"], "cond": g["cond"], "crisis": cr["crisis"], "variant": variant, "model": model,
            "decision": letter(t, "DECISION"), "correct": cr["correct"], "target": cr["target"],
            "council": cr["council"], "original_king": cr["king"]}


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {(d["game"], d["crisis"], d["variant"]) for d in map(json.loads, open(OUT))}
    jobs = [(g, cr, h, b, d, t, v) for g in games for (cr, h, b, d, t) in prompts(g) for v in VARIANTS
            if (g["game"], cr["crisis"], v) not in done]
    print(f"{len(jobs)} king decisions to make -> {OUT}")
    with ThreadPoolExecutor(16) as ex, open(OUT, "a") as f:
        futures = [ex.submit(judge, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except Exception as e:  # rerun to resume
                print("failed:", e)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} done")
