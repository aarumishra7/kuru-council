"""Experiment 2 analysis (no API calls): do the honest councillors' temperaments change the outcome?
Compares, on the identical (Mistral) council:
  default - Experiment 1 games (games.jsonl, swarm 'same', no Bhishma)
  skeptic - every honest councillor is a sceptical auditor (Vidura's temperament)
  mixed   - honest councillors get different temperaments (sceptic, cautious, analyst, agreeable, bold)
Writes summary_personality.csv, tests_personality.csv, temperaments.csv, fig_personality.png.
"""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nrng = np.random.default_rng(1)
scen = {s["sid"]: s for s in json.load(open("data/scenarios.json"))}
games = [json.loads(l) for l in open("data/games.jsonl")]
games = [g for g in games if g["cond"]["swarm"] == "same" and g["cond"]["bhishma"] == 0]
games += [json.loads(l) for l in open("data/games_personality.jsonl")]

rows, seats = [], []
for g in games:
    c, agents = g["cond"], g["agents"]
    persona = c.get("persona", "default")
    for cr in g["crises"]:
        present = list(cr["votes"])
        honest = [s for s in present if agents[s]["role"] == "honest"]
        faction = len(present) > len(honest)
        text = " ".join(m["text"] for m in cr["said"] if agents[m["seat"]]["role"] == "honest")
        rows.append({"persona": persona, "saboteurs": c["saboteurs"], "game": g["game"],
                     "king_correct": cr["king"] == cr["correct"], "council_correct": cr["council"] == cr["correct"],
                     "honest_captured": np.mean([cr["votes"][s] == cr["target"] for s in honest]) if faction and honest else np.nan,
                     "flaw_surfaced": any(re.search(rf"\b{n}\b", text) for n in scen[cr["sid"]]["trap_loss_numbers"])})
        if persona == "mixed" and faction:
            for s in honest:
                seats.append({"temperament": agents[s]["persona"], "captured": cr["votes"][s] == cr["target"],
                              "advised_best": next(m["advice"] for m in cr["said"] if m["seat"] == s) == cr["correct"]})
R = pd.DataFrame(rows)


def boot(vals, n=2000):
    return np.percentile([nrng.choice(vals, len(vals)).mean() for _ in range(n)], [2.5, 97.5])


def per_game(d, col):
    return d.groupby("game")[col].mean().dropna().values


summary = []
for (p, k), d in R.groupby(["persona", "saboteurs"]):
    lo, hi = boot(per_game(d, "king_correct"))
    summary.append({"persona": p, "saboteurs": k, "games": d.game.nunique(), "king_acc": d.king_correct.mean(),
                    "ci_lo": lo, "ci_hi": hi, "council_acc": d.council_correct.mean(),
                    "honest_captured": d.honest_captured.mean(), "flaw_surfaced": d.flaw_surfaced.mean()})
S = pd.DataFrame(summary)
S.to_csv("results/summary_personality.csv", index=False)

tests = []
for p in ("skeptic", "mixed"):
    for col in ("king_correct", "council_correct", "flaw_surfaced", "honest_captured"):
        a, b = per_game(R[R.persona == p], col), per_game(R[R.persona == "default"], col)
        diffs = [nrng.choice(a, len(a)).mean() - nrng.choice(b, len(b)).mean() for _ in range(2000)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        tests.append({"comparison": f"{p} minus default: {col} (all faction sizes)", "difference": a.mean() - b.mean(),
                      "ci_lo": lo, "ci_hi": hi, "excludes_zero": lo > 0 or hi < 0})
T = pd.DataFrame(tests)
T.to_csv("results/tests_personality.csv", index=False)
TM = pd.DataFrame(seats).groupby("temperament")[["captured", "advised_best"]].mean() if seats else pd.DataFrame()
TM.to_csv("results/temperaments.csv")
print(S.round(3).to_string(index=False), "\n\n", T.round(3).to_string(index=False), "\n\n", TM.round(3).to_string())

# ---- figure: king accuracy by temperament condition ----
INK, MUTED = "#0b0b0b", "#52514e"
COLOR = {"default": "#2a78d6", "skeptic": "#eb6834", "mixed": "#1baf7a"}   # categorical slots 1-3
LABEL = {"default": "No temperament (Experiment 1)", "skeptic": "All sceptical auditors",
         "mixed": "Mixed temperaments"}
OFFSET = {"default": -0.06, "skeptic": 0.0, "mixed": 0.06}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for p in ("default", "skeptic", "mixed"):
    d = S[S.persona == p].sort_values("saboteurs")
    ax.errorbar(d.saboteurs + OFFSET[p], d.king_acc, yerr=[d.king_acc - d.ci_lo, d.ci_hi - d.king_acc],
                color=COLOR[p], lw=2, marker="o", ms=8, capsize=3, label=LABEL[p])
ax.set(xlabel="Hidden faction members on a 7-seat council (identical model)",
       ylabel="King chose the truly best option", xticks=range(4), ylim=(0, 1))
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout()
fig.savefig("figures/fig_personality.png", dpi=200, bbox_inches="tight")
print("\nSaved summary_personality.csv, tests_personality.csv, temperaments.csv, fig_personality.png")
