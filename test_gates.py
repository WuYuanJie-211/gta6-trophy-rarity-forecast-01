#!/usr/bin/env python3
"""Gate checks for the trophy-rarity forecast entry (NOT part of the entry).

Gate 1: reproduce the house persistence baseline exactly + verify the pinball
scorer against the rules' worked example (loss 0.403404).
Gate 2: when the history has a player-base jump, the N/E decomposition must
naturally reproduce the diluted answer (synthetic jump test).
Gate 3: format guards (two runs byte-identical, parseable, monotone, <8s,
<=65,536 B) — run separately in the shell.

The practice answer is NOT in the kit; the synthetic answer below reconstructs
it from the rules' description of the 04:30 refresh (3,868 players tracked,
every rate but the 100% one lower): earners held at their last-read values,
rates rescaled by 638/3868. It is a proxy for scorer-scale validation only.
"""
import json
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01")
import main as M

KIT = "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/gta6-trophy-rarity-forecast-01-kit"
REPO = "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01"


def load_case(cid):
    with open(f"{KIT}/input/{cid}.in") as f:
        return json.load(f)


def load_out(path):
    with open(path) as f:
        d = json.load(f)
    return {r["id"]: r["q"] for r in d["trophies"]}


# ---- Gate 1a: house baseline reproduction ---------------------------------
case = load_case("p00")
mine = M.house_baseline(case)
starter = load_out(f"{KIT}/out/p00.out")
maxdiff = 0.0
for tid in mine:
    for a, b in zip(mine[tid], starter[tid]):
        maxdiff = max(maxdiff, abs(a - b))
print(f"Gate 1a baseline repro: max |mine - starter| = {maxdiff:.6f} "
      f"({'PASS' if maxdiff < 1e-9 else 'FAIL'})")

# ---- Gate 1b: pinball scorer vs the rules' worked example ------------------
rows = {1: [10, 15, 20, 25, 35], 2: [10, 15, 20, 25, 35]}
answers = {1: 22.0, 2: 8.0}
loss = M.pinball_loss(rows, answers)
print(f"Gate 1b pinball worked example: loss = {loss:.6f} (expect 0.403404) "
      f"({'PASS' if abs(loss - 0.403404) < 1e-6 else 'FAIL'})")

# ---- Practice proxy: house baseline vs my forecast -------------------------
# Synthetic answer: earners held, rates rescaled 638 -> 3868 players.
last = case["history"][-1]["rates"]
synth = {}
for t, r in zip(case["trophies"], last):
    synth[t["id"]] = 100.0 if r >= 99.99 else r * 638.0 / 3868.0
house_loss = M.pinball_loss(starter, synth)
mine_out = load_out(f"{REPO}/out1/p00.out")
my_loss = M.pinball_loss(mine_out, synth)
print(f"Practice proxy (synthetic diluted answer, 638->3868 players):")
print(f"  house baseline loss = {house_loss:.4f} (rules say ~1.51)")
print(f"  my forecast loss    = {my_loss:.4f}")
print(f"  improvement         = {house_loss - my_loss:+.4f} "
      f"({'PASS: beats house on proxy' if my_loss < house_loss else 'FAIL'})")
# Typical miss + 90% coverage for reference.
miss = sum(abs(synth[i] - mine_out[i][2]) for i in mine_out) / 55
cov = sum(1 for i in mine_out if mine_out[i][0] <= synth[i] <= mine_out[i][4]) / 55
print(f"  my typical miss = {miss:.3f} pts, 90% coverage = {cov:.2f}")

# ---- Gate 2: synthetic player-base jump ------------------------------------
# Build a case whose history ends with a jump: same rates but N 638 -> 3868
# at the last read (earners rescaled so rates stay the same), then a further
# jump is NOT in the history. Instead, test the mechanism directly: append a
# read where N jumps 6x and E stays constant, so rates dilute 6x; the
# decomposition (E constant, N jumped) must forecast the diluted rate.
import copy
case2 = copy.deepcopy(case)
h = copy.deepcopy(case["history"][-1])
h["at"] = "2026-11-19T03:00:00Z"
h["slot"] = "2026-11-19T03:00:00Z"
h["players_tracked"] = 3868
# Earners unchanged -> rates scale by 638/3868.
h["rates"] = [100.0 if r >= 99.99 else r * 638.0 / 3868.0
              for r in h["rates"]]
case2["history"].append(h)
rows2 = M.forecast_case(case2)
# The forecast center for a 1.1% trophy should be near the diluted rate
# (0.18%), not the original 1.1%: the decomposition sees E flat and N jumped.
c4 = rows2[4][2]
print(f"Gate 2 jump test: trophy 4 median = {c4:.4f}% (diluted answer ~0.1813%, "
      f"last pre-jump rate 1.1%)")
print(f"  {'PASS' if abs(c4 - 0.1813) < 0.25 else 'FAIL'}: "
      f"decomposition {'reproduces' if abs(c4 - 0.1813) < 0.25 else 'does NOT reproduce'} "
      f"the dilution")
