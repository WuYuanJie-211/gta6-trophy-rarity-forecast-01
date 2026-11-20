#!/usr/bin/env python3
"""Stress test: synthetic final-like case (rich history, growing N, real trends).

Builds a case with ~3 days of reads (every 30 min), N growing 638 -> ~50k,
early trophies' earners rising fast, grind trophies flat. Runs main.py on it
and checks the output is valid and the centers are sensible. This exercises
the paths the flat practice cases never touch (g0>0, med_de>0, gamma>0,
ceilings binding, AR blend).
"""
import json
import math
import os
import subprocess
import sys
import time

KIT = "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/gta6-trophy-rarity-forecast-01-kit"
REPO = "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01"

with open(f"{KIT}/input/p00.in") as f:
    base = json.load(f)

# --- synthetic history: 3 days, every 30 min (145 reads) --------------------
# N: log growth, fast early then slowing (damped): N(t) = 638 * exp(0.9*(1-exp(-t/48)))
# E_i: per-trophy earners grow toward a category ceiling fraction of N.
import datetime
t0 = datetime.datetime(2026, 11, 18, 23, 50, tzinfo=datetime.timezone.utc)
reads = []
# Category ceiling fractions (same as the model's priors, for realism).
ceil = {}
for t in base["trophies"]:
    d = t["description"]
    if "Gone Fishing" in d:
        ceil[t["id"]] = 1.0
    elif t["tier"] == "platinum" or "100% Completion" in d:
        ceil[t["id"]] = 0.02
    elif "Chapter 1" in d or "Chapter 2" in d:
        ceil[t["id"]] = 0.60
    elif "Chapter 3" in d or "Chapter 4" in d or d.strip() == "Complete the final mission.":
        ceil[t["id"]] = 0.25
    elif any(k in d for k in ["250 Stunts", "gold medal", "Drone Race", "20 different races",
                              "8 mission strands", "Meditate", "Collectible Card",
                              "100 locations", "50 Challenges", "100 vehicles", "5 Bank Vaults"]):
        ceil[t["id"]] = 0.05
    elif any(k in d for k in ["laundering", "Spend $", "garage properties", "Marked Bills"]):
        ceil[t["id"]] = 0.10
    else:
        ceil[t["id"]] = 0.25
# Earners approach the ceiling with a per-trophy speed (early trophies fast).
speed = {}
for t in base["trophies"]:
    d = t["description"]
    if "Gone Fishing" in d:
        speed[t["id"]] = 99.0
    elif "Chapter 1" in d or "Chapter 2" in d:
        speed[t["id"]] = 0.10
    elif "Chapter 3" in d or "Chapter 4" in d:
        speed[t["id"]] = 0.05
    else:
        speed[t["id"]] = 0.02

n_reads = 145
for k in range(n_reads):
    t = t0 + datetime.timedelta(minutes=30 * k)
    hours = k * 0.5
    n = 638.0 * math.exp(0.9 * (1.0 - math.exp(-hours / 48.0)))
    # One refresh jump at read 100 (N jumps 1.4x at once).
    if k >= 100:
        n *= 1.4
    rates = []
    for tr in base["trophies"]:
        c = ceil[tr["id"]]
        s = speed[tr["id"]]
        # Earners fraction of N approaches c with speed s (per hour).
        frac = c * (1.0 - math.exp(-s * hours)) if s < 10 else c
        # Launch-night floor: match the observed last rates at k=0.
        rates.append(round(max(frac, 0.0063) * 100.0, 2))
    # Overwrite the first read's rates with the real launch-night rates.
    if k == 0:
        rates = list(base["history"][-1]["rates"])
    reads.append({
        "slot": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "at": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "players_tracked": int(round(n)),
        "trophies_earned": int(sum(r * n / 100.0 for r in rates)),
        "rates": rates,
    })

case = dict(base)
case["id"] = "f00synth"
case["at"] = "2026-11-21T11:30:00Z"
case["horizon_minutes"] = 6510
case["target"] = "2026-11-26T00:00:00Z"
case["history"] = reads
os.makedirs("/tmp/synth_in", exist_ok=True)
with open("/tmp/synth_in/f00synth.in", "w") as f:
    json.dump(case, f)
print(f"synthetic case: {len(reads)} reads, N {reads[0]['players_tracked']} -> "
      f"{reads[-1]['players_tracked']}, last rates[0:5] = {reads[-1]['rates'][:5]}")

os.makedirs("/tmp/synth_out", exist_ok=True)
t_start = time.time()
r = subprocess.run([sys.executable, f"{REPO}/main.py"], env={
    **os.environ, "MAKEFUN_INPUT_DIR": "/tmp/synth_in",
    "MAKEFUN_OUTPUT_DIR": "/tmp/synth_out"}, capture_output=True, text=True)
dt = time.time() - t_start
print(f"run: rc={r.returncode}, {dt:.2f}s {'PASS' if r.returncode == 0 and dt < 8 else 'FAIL'}")
if r.returncode != 0:
    print(r.stderr[-2000:])
    sys.exit(1)
with open("/tmp/synth_out/f00synth.out") as f:
    d = json.load(f)
rows = {x["id"]: x["q"] for x in d["trophies"]}
size = os.path.getsize("/tmp/synth_out/f00synth.out")
mono = all(q == sorted(q) for q in rows.values())
print(f"output: {len(rows)} rows, {size} bytes (<=65536: {size <= 65536}), monotone: {mono}")
names = {t["id"]: t["name"] for t in base["trophies"]}
for tid in [1, 2, 3, 4, 5, 7, 14, 15, 25, 28, 35, 40, 52]:
    print(f"  id={tid:2d} {names[tid][:28]:28s} median={rows[tid][2]:7.4f}%  "
          f"q05={rows[tid][0]:7.4f} q95={rows[tid][4]:7.4f}")
# Sanity: Ch1 median should be high (rising), grind/platinum low, saturated ~100.
ok = (rows[3][2] > 20 and rows[15][2] < 8 and rows[1][2] < 3
      and rows[2][2] > 99 and rows[4][2] > rows[5][2])
print("sanity (Ch1>20, grind<8, plat<3, saturated>99, Ch2>Ch3):",
      "PASS" if ok else "FAIL")
