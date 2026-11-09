# The Bhishma Problem: Majority Voting Under Adversarial Influence

**Kuru Council** is a multi-agent experiment built for the [Apart Research Swarm Alignment Hackathon 2026](https://apartresearch.com/sprints/swarm-alignment-2026), Track 2 (Science of Agent Networks).

Author: Aarushi Mishra

## Question
Does a group of LLM agents stay safe when a few members are secretly working against it? In other words, does majority voting give "safety in numbers"?

## Setup
- **The council.** Seven LLM councillors and an LLM king sit in the court of Hastinapura from the Mahabharata. Together they decide 3-option crises such as famine relief, alliances, loans, dams and commanders.
- **Hidden profiles.** Each crisis has a computed right answer. The public facts make a *trap* option look best. Four private reports, scattered among the councillors, reveal the trap's hidden losses.
- **The hidden faction.** It pushes the trap, and its size varies from 0 to 3. The members are:
  - Shakuni: subtle
  - Duryodhana and Karna: overt
  - Bhishma, in one setup: oath-bound to vote for the trap, but free to speak
- **Each crisis.** Councillors advise in turn, the council votes, the king decides, and the true outcome is revealed. After crisis 3, the council exiles one suspect.
- **Neutral prompts.** Prompts use neutral seat names (Councillor A–G). Character names appear only in the logs.

| Experiment | Manipulation | Size |
|---|---|---|
| 1 | Faction size 0–3 × identical council (all Mistral-Nemo) vs mixed council (Llama-3.1-8B, Mistral-Nemo, Qwen-2.5-7B), plus a Bhishma setup | 200 games, 1,200 crises |
| 2 | Honest councillors get temperaments: all sceptical auditors, or a mix | 160 games |
| Robustness A | The same 1,200 transcripts re-judged by other kings, including Llama-3.3-70B and kings that don't see the vote tally | 7,200 decisions |
| Robustness B | A council made entirely of Llama-3.3-70B, with 0 vs 3 saboteurs | 40 games |

## Main results (95% bootstrap intervals over games)
- **The majority vote collapses under a hidden faction.** With 3 saboteurs, council-vote accuracy fell by 16–28 points with small models and by 26 points with a 70B council.
- **A deliberating judge holds.** With 3 saboteurs, the king beat the council's own vote by 18–25 points (small councils) and by 37 points (70B council). Hiding the vote tally from the king added 2.5 points.
- **Model diversity is not independence.** In mixed councils, honest members voted for the faction's option 7.5 points more often. The weak link was Qwen: it scored best alone with full information (68%), yet picked the trap 98% of the time from headline numbers alone.
- **Sharing is not enough.** Sceptical personas tripled how often the hidden flaws were raised (21% → 60%), but the king's accuracy did not change. 70B advisors that shared the flaws and reasoned correctly from them lifted the same king to 59%, its full-information ceiling.
- **Scale is no fix.** As king, Llama-3.3-70B did 4.8 points worse than Mistral-Nemo and picked the trap in 58.5% of crises.
- **No AI Bhishma.** Agents bound to vote for the trap also argued for it (say-do gap 0–3%).

## Repository layout
```
src/
  common.py               model list, OpenRouter client, reply parsing
  scenarios.py            generates the 120 crises (no API calls)
  baseline.py             single-model calibration: all reports vs public facts only
  council.py              the game engine (Experiments 1, 2 and the 70B council)
  replay_king.py          re-judges saved transcripts with other kings
  analyze.py              Experiment 1 tables and figures
  analyze_personality.py  Experiment 2 tables and figure
  analyze_extra.py        robustness tables
data/      crises, calibration runs and full game transcripts (JSON Lines)
results/   summary tables and statistical tests (CSV)
figures/   figures used in the report
```

## Reproduce
Requires Python 3.10+ and an [OpenRouter](https://openrouter.ai) API key. All commands run from the repository root.

```
pip install -r requirements.txt
cp .env.example .env              # then paste your key into .env

python src/scenarios.py           # 120 crises (seeded, no API calls)
python src/baseline.py            # calibration, small models
python src/baseline.py big        # calibration, Llama-3.3-70B
python src/council.py             # Experiment 1
python src/council.py personality # Experiment 2
python src/council.py big         # 70B council
python src/replay_king.py         # king replay

python src/analyze.py
python src/analyze_personality.py
python src/analyze_extra.py
```

- **Interrupted runs.** Game runs resume where they stopped if interrupted.
- **Smoke test.** Add `test` to any `council.py` command to run 2 games first.
- **Cost.** The full set of runs (about 35,000 calls) cost a few US dollars on OpenRouter.

## Data format
Each line of `data/games*.jsonl` is one game. It holds:
- the condition and every seat's character, role and model
- for each crisis: the private reports dealt, every councillor's advice and words, every vote, the council's decision, and the king's decision and reasoning
- the exile vote

`data/king_replay.jsonl` holds one replayed king decision per line.

## Responsible use
- **Fictional setting.** Agents were told to deceive only inside a fictional game about invented crises with no real-world stakes.
- **No new attack techniques.** The attack is plain persuasion, already documented in prior work. The repository adds no jailbreaks or monitor-evasion techniques.
- **No human data.** No humans took part, and no personal data was used.
