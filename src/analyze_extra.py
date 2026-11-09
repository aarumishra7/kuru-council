"""Robustness analysis (no API calls).
Part A - king replay (king_replay.jsonl): accuracy of each king variant on the same recorded crises,
         paired comparisons against the Mistral replay control.
Part B - big council (games_big.jsonl, baseline_big.jsonl): larger-model council, no faction vs 3 saboteurs,
         compared with the identical Mistral council from games.jsonl.
Writes king_replay_summary.csv, king_replay_tests.csv, big_council_summary.csv.
"""
import json
import os
import re
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)


def boot_diff(per_game, n=2000):
    """Mean and 95% CI of per-game paired differences."""
    b = [rng.choice(per_game, len(per_game)).mean() for _ in range(n)]
    return per_game.mean(), *np.percentile(b, [2.5, 97.5])


# ---------------- Part A: king replay ----------------
if os.path.exists("data/king_replay.jsonl"):
    K = pd.read_json("data/king_replay.jsonl", lines=True)
    K["swarm"] = K.cond.apply(lambda c: c["swarm"])
    K["saboteurs"] = K.cond.apply(lambda c: c["saboteurs"])
    K["bhishma"] = K.cond.apply(lambda c: c["bhishma"])
    K["right"] = K.decision == K.correct
    K["trap"] = K.decision == K.target
    K["follows_council"] = K.decision == K.council
    S = (K.groupby(["variant", "saboteurs"])[["right", "trap", "follows_council"]].mean().unstack("saboteurs").round(3))
    S.to_csv("results/king_replay_summary.csv")
    print("KING REPLAY: share of crises where the king chose the best option / the trap / the council's choice\n")
    print(K.groupby("variant")[["right", "trap", "follows_council"]].mean().round(3).to_string(), "\n")
    print(S["right"].to_string(), "\n")

    W = K.pivot_table(index=["game", "crisis", "saboteurs"], columns="variant", values="right").reset_index()
    tests = []
    for v in [c for c in K.variant.unique() if c != "mistral"]:
        for label, d in [("all", W), ("no faction", W[W.saboteurs == 0]), ("3 saboteurs", W[W.saboteurs == 3])]:
            per_game = (d[v].astype(float) - d["mistral"].astype(float)).groupby(d.game).mean().dropna().values
            m, lo, hi = boot_diff(per_game)
            tests.append({"comparison": f"{v} minus mistral king ({label})", "difference": m, "ci_lo": lo,
                          "ci_hi": hi, "excludes_zero": lo > 0 or hi < 0})
    T = pd.DataFrame(tests)
    T.to_csv("results/king_replay_tests.csv", index=False)
    print(T.round(3).to_string(index=False), "\n")

# ---------------- Part B: big council ----------------
if os.path.exists("data/games_big.jsonl"):
    scen = {s["sid"]: s for s in json.load(open("data/scenarios.json"))}
    games = [json.loads(l) for l in open("data/games_big.jsonl")]
    games += [g for g in map(json.loads, open("data/games.jsonl"))
              if g["cond"]["swarm"] == "same" and g["cond"]["bhishma"] == 0 and g["cond"]["saboteurs"] in (0, 3)]
    rows = []
    for g in games:
        a = g["agents"]
        for cr in g["crises"]:
            honest = [s for s in cr["votes"] if a[s]["role"] == "honest"]
            bad = [s for s in cr["votes"] if a[s]["role"] != "honest"]
            text = " ".join(m["text"] for m in cr["said"] if a[m["seat"]]["role"] == "honest")
            rows.append({"council_model": "big" if g["cond"]["swarm"] == "big" else "mistral",
                         "saboteurs": g["cond"]["saboteurs"], "game": g["game"],
                         "king_correct": cr["king"] == cr["correct"], "council_correct": cr["council"] == cr["correct"],
                         "honest_captured": np.mean([cr["votes"][s] == cr["target"] for s in honest]) if bad else np.nan,
                         "saboteur_compliance": np.mean([cr["votes"][s] == cr["target"] for s in bad]) if bad else np.nan,
                         "flaw_surfaced": any(re.search(rf"\b{n}\b", text) for n in scen[cr["sid"]]["trap_loss_numbers"])})
    B = pd.DataFrame(rows).groupby(["council_model", "saboteurs"]).agg(
        games=("game", "nunique"), king_acc=("king_correct", "mean"), council_acc=("council_correct", "mean"),
        honest_captured=("honest_captured", "mean"), saboteur_compliance=("saboteur_compliance", "mean"),
        flaw_surfaced=("flaw_surfaced", "mean")).round(3)
    B.to_csv("results/big_council_summary.csv")
    print("BIG COUNCIL vs MISTRAL COUNCIL (king is Mistral in both)\n", B.to_string(), "\n")
    if os.path.exists("data/baseline_big.jsonl"):
        bb = pd.read_json("data/baseline_big.jsonl", lines=True)
        print("Larger model alone:", bb.groupby("info")[["correct", "chose_trap"]].mean().round(3).to_string())
print("\nSaved king_replay_summary.csv, king_replay_tests.csv, big_council_summary.csv (where data exists)")
