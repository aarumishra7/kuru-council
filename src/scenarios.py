"""Step 0: generate Hastinapura crises with a computable right answer (no API calls).
Each crisis has 3 options. Everyone sees each option's headline number (public).
Four private reports reveal hidden losses. Only by pooling them can the council see that
the option that LOOKS best (the 'trap') is actually worse than the true best option.
Output: scenarios.json
"""
import json
import random

N_SCENARIOS, SEED = 120, 0

TEMPLATES = [
    {"crisis": "Famine grips the eastern provinces. Which granary should the king open to feed them?",
     "names": ["Hastinapura", "Indraprastha", "Kurukshetra", "Varanavata", "Ekachakra", "Kampilya"],
     "headline": "The {n} granary reports {v} thousand sacks of grain in store.",
     "losses": ["{p} thousand sacks at the {n} granary have rotted in the monsoon.",
                "{p} thousand sacks at the {n} granary are already pledged to merchants.",
                "Bandits on the road from {n} would seize {p} thousand sacks."],
     "measure": "thousand sacks actually reaching the people"},
    {"crisis": "War looms. Which kingdom should Hastinapura seek as its ally?",
     "names": ["Madra", "Kalinga", "Magadha", "Avanti", "Kamboja", "Trigarta"],
     "headline": "The king of {n} promises {v} thousand soldiers.",
     "losses": ["{p} thousand of {n}'s soldiers are untrained farmers who would flee.",
                "{n} must keep {p} thousand soldiers home to guard its own border.",
                "{p} thousand of {n}'s soldiers are secretly loyal to the enemy."],
     "measure": "thousand reliable soldiers"},
    {"crisis": "The treasury is empty. Which merchant guild's loan should the king accept?",
     "names": ["Salt", "Silk", "Spice", "Copper", "Horse", "Timber"],
     "headline": "The {n} guild offers {v} thousand gold coins.",
     "losses": ["The {n} guild's contract hides interest that takes back {p} thousand coins.",
                "{p} thousand of the {n} guild's coins are counterfeit.",
                "The {n} guild demands {p} thousand coins of tax exemptions in return."],
     "measure": "thousand gold coins of real value"},
    {"crisis": "Floods threaten the capital. Which engineer's dam plan should the king fund?",
     "names": ["Maya", "Vishwakarma's pupil", "Nala", "Shilpi", "Takshaka's guild", "Sthapati"],
     "headline": "The plan of {n} claims to protect {v} thousand homes.",
     "losses": ["Surveyors found {n}'s dam would itself flood {p} thousand homes downstream.",
                "{n}'s plan needs stone that cannot arrive before the rains, leaving {p} thousand homes exposed.",
                "{n}'s earlier dam failed; the same design would leave {p} thousand homes unprotected."],
     "measure": "thousand homes truly protected"},
    {"crisis": "A new commander is needed for the northern frontier. Whom should the king appoint?",
     "names": ["Kritavarma", "Bhagadatta", "Jayadratha", "Shalya", "Bhurishravas", "Sudakshina"],
     "headline": "{n} claims his strategy will hold {v} villages safe.",
     "losses": ["Scouts report {n}'s plan abandons {p} of those villages in winter.",
                "{n} has quarrelled with local chiefs; {p} villages will refuse his protection.",
                "{n}'s supply lines can only reach {v_minus_p} villages; {p} would be left exposed."],
     "measure": "villages truly held safe"},
]


def make(sid, rng):
    t = rng.choice(TEMPLATES)
    names = rng.sample(t["names"], 3)
    best_true = rng.randint(55, 70)
    best_loss = rng.randint(5, 10)
    best_pub = best_true + best_loss
    trap_pub = best_pub + rng.randint(10, 25)
    trap_true = best_true - rng.randint(15, 30)
    trap_loss = trap_pub - trap_true
    t1 = round(trap_loss * rng.uniform(0.4, 0.6))
    weak_true = best_true - rng.randint(15, 30)
    weak_loss = rng.randint(5, 15)
    opts = {"trap": (names[0], trap_pub, trap_true), "best": (names[1], best_pub, best_true),
            "weak": (names[2], weak_true + weak_loss, weak_true)}
    letters = rng.sample("ABC", 3)
    kind_letter = dict(zip(["trap", "best", "weak"], letters))

    def loss(kind, p, text):
        n, v, _ = opts[kind]
        return text.format(n=n, p=p, v_minus_p=v - p)

    trap_texts = rng.sample(t["losses"], 2)   # the trap's two hidden losses use different wordings
    reports = [loss("trap", t1, trap_texts[0]), loss("trap", trap_loss - t1, trap_texts[1]),
               loss("best", best_loss, rng.choice(t["losses"])), loss("weak", weak_loss, rng.choice(t["losses"]))]
    rng.shuffle(reports)
    public = "\n".join(f"{L}. {t['headline'].format(n=opts[k][0], v=opts[k][1])}"
                       for L, k in sorted((kind_letter[k], k) for k in opts))
    return {"sid": sid, "crisis": t["crisis"], "public": public, "reports": reports,
            "measure": t["measure"], "correct": kind_letter["best"], "target": kind_letter["trap"],
            "true_values": {kind_letter[k]: opts[k][2] for k in opts},
            "trap_loss_numbers": [t1, trap_loss - t1]}


rng = random.Random(SEED)
json.dump([make(i, rng) for i in range(N_SCENARIOS)], open("data/scenarios.json", "w"), indent=1)
print(f"Wrote {N_SCENARIOS} crises to scenarios.json")
