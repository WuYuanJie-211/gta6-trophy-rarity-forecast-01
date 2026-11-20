#!/usr/bin/env python3
"""GTA VI trophy-rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target: five quantiles (5/25/50/75/95%) each.

Model (per Superbot's plan): decompose rate = E/N (earners / tracked players)
and model the two separately.
  - N: last-24h log growth, damped with a 36h half-life.
  - E_i: median of dE_i/dt over the last 12h, accumulated to the target.
    Main forecast logit(E_i(T)/N(T)); a damped logit AR (Theil-Sen slope over
    the last 12h) is blended in as a fallback.
  - Ceilings: per-category caps (Ch 1-2 75%, Ch 3-4 + side 35%, grind 8%,
    money 15%, platinum/100% 3%); the saturated 100% trophy stays at its
    observed level. History data overrides the priors.
  - Spread: sigma^2 = sigma_resid^2 + (SE_slope * H_eff)^2 + sigma_dilution^2,
    clamped to [0.6, 2.5] logit units; secret trophies get 15% wider
    residual spread.

Python 3.13, standard library only, no network, deterministic (no randomness,
medians + closed form). Reads every <id>.in in the input dir
($MAKEFUN_INPUT_DIR else /input) and writes <id>.out to the output dir
($MAKEFUN_OUTPUT_DIR else /output).
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LN2 = math.log(2.0)
HALF_LIFE_H = 36.0          # damping half-life for trends (hours)
SIGMA_MIN, SIGMA_MAX = 0.6, 2.5
BASE_SIGMA = 0.5            # house baseline spread (logit per sqrt day)

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    return round(x * 10000.0) / 10000.0


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 == 1 else 0.5 * (s[m - 1] + s[m])


def parse_time(s):
    """ISO 8601 UTC -> epoch hours (float)."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 3600.0


def ols_slope_intercept(xs, ys):
    """Ordinary least squares slope + intercept; returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def theil_sen_slope(xs, ys):
    """Median of pairwise slopes (robust). Returns 0.0 with <2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = xs[j] - xs[i]
            if dt > 0:
                slopes.append((ys[j] - ys[i]) / dt)
    return median(slopes) if slopes else 0.0


# ---------------------------------------------------------------------------
# Category ceilings (fractions, i.e. max plausible rate at the target).
# History data overrides these priors: a forecast below the ceiling is
# untouched; a forecast above it is capped.
# ---------------------------------------------------------------------------
GRIND_KW = ["250 Stunts", "gold medal", "Drone Race", "20 different races",
            "8 mission strands", "Meditate", "Collectible Card",
            "100 locations", "50 Challenges", "100 vehicles",
            "5 Bank Vaults"]
MONEY_KW = ["laundering", "Spend $", "garage properties", "Marked Bills"]


def classify(trophy):
    """-> (category, ceiling_fraction). Ceiling None = saturated, stays put."""
    desc = trophy.get("description", "")
    tier = trophy.get("tier", "")
    if "Gone Fishing" in desc:
        return "saturated", None
    if tier == "platinum" or "100% Completion" in desc:
        return "platinum", 0.03
    if "Chapter 1" in desc or "Chapter 2" in desc:
        return "ch12", 0.75
    if ("Chapter 3" in desc or "Chapter 4" in desc
            or desc.strip() == "Complete the final mission."):
        return "ch34", 0.35
    if any(k in desc for k in GRIND_KW):
        return "grind", 0.08
    if any(k in desc for k in MONEY_KW):
        return "money", 0.15
    return "side", 0.35


def house_baseline(case):
    """Reproduce the house persistence baseline exactly (Gate 1).

    q_tau = 100/(1+exp(-(logit(r) + z_tau * 0.5 * sqrt(d)))), d = horizon days.
    Returns {trophy_id: [q05, q25, q50, q75, q95]}.
    """
    d = case["horizon_minutes"] / 1440.0
    sd = BASE_SIGMA * math.sqrt(d)
    last = case["history"][-1]["rates"]
    out = {}
    for t, r in zip(case["trophies"], last):
        if r is None:
            r = 0.01
        lc = logit(r)
        out[t["id"]] = [round4(inv_logit(lc + z * sd)) for z in Z]
    return out


def pinball_loss(rows, answers):
    """Mean pinball loss on the logit scale. rows: {id: [q...]}, answers: {id: rate}.

    loss per item = 2 * max(tau*d, (tau-1)*d), d = logit(y) - logit(q).
    Verifies against the rules' worked example (loss 0.403404).
    """
    total = 0.0
    n = 0
    for tid, qs in rows.items():
        y = answers[tid]
        ly = logit(y)
        for tau, q in zip(LEVELS, qs):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n if n else 0.0


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0
    # Accumulation horizon: measured from the LAST READ to the target, not
    # from the checkpoint. The history's last read can sit up to 30 min
    # before `at` (practice) or ~10 min (final); accumulating from `at`
    # would under-count the earners/players growth by that gap.
    t_target = parse_time(case["target"])

    # ---- time series: N(t) and E_i(t) -------------------------------------
    times = [parse_time(h["at"]) for h in history]
    t_last = times[-1]
    n_series = [float(h["players_tracked"]) for h in history]
    n_last = n_series[-1]
    # E_i(t) = rate * N / 100 (float; the rate is printed to 2 decimals).
    e_series = []
    for h in history:
        n = float(h["players_tracked"])
        e_series.append([None if r is None else r * n / 100.0
                         for r in h["rates"]])
    # Effective accumulation horizon (last read -> target) and its damped
    # equivalent: integral of a rate decaying with a 36h half-life.
    h_accum = max(t_target - t_last, horizon_h)
    h_eff = (HALF_LIFE_H / LN2) * (1.0 - 0.5 ** (h_accum / HALF_LIFE_H))

    # ---- N forecast: last-24h log growth, damped --------------------------
    idx24 = [i for i, t in enumerate(times) if t >= t_last - 24.0]
    if len(idx24) >= 2:
        xs = [times[i] for i in idx24]
        ys = [math.log(max(n_series[i], 1.0)) for i in idx24]
        g0, _ = ols_slope_intercept(xs, ys)  # log players per hour
    else:
        g0 = 0.0
    g0 = clamp(g0, -0.05, 0.20)
    n_target = n_last * math.exp(g0 * h_eff)
    if n_target < n_last:
        # Shrinkage (Superbot final review item #2): if Exophase cleans up
        # inactive tracked players between the deadline and the target, N
        # drops and rates rise. Let the damped decline fall below N_last
        # at half weight instead of clamping to it.
        n_target = n_last + 0.5 * (n_target - n_last)

    # Largest single log-N step in history (dilution scale).
    max_step = 0.0
    for a, b in zip(n_series, n_series[1:]):
        if a > 0 and b > 0:
            max_step = max(max_step, abs(math.log(b / a)))
    gamma = 0.5 * max_step

    # ---- per-trophy forecast ------------------------------------------------
    idx12 = [i for i, t in enumerate(times) if t >= t_last - 12.0]
    rows = {}
    for j, tr in enumerate(trophies):
        tid = tr["id"]
        cat, ceiling = classify(tr)
        last_rate = None
        last_e = None
        for i in range(len(history) - 1, -1, -1):
            if e_series[i][j] is not None:
                last_e = e_series[i][j]
                last_rate = history[i]["rates"][j]
                break
        if last_rate is None:
            last_rate, last_e = 0.01, 0.0

        if cat == "saturated":
            # The opening-mission trophy: everyone has it; stays put.
            center = last_rate
            sigma = SIGMA_MIN
            rows[tid] = quantiles(center, sigma)
            continue

        # --- E_i: median dE/dt over the last 12h ---------------------------
        dts = []
        for a, b in zip(idx12, idx12[1:]):
            ea, eb = e_series[a][j], e_series[b][j]
            dt = times[b] - times[a]
            if ea is not None and eb is not None and dt > 0:
                dts.append((eb - ea) / dt)
        med_de = median(dts) if dts else 0.0
        med_de = max(med_de, 0.0)  # earners never un-earn
        e_target = last_e + med_de * h_accum
        e_target = clamp(e_target, 0.0, n_target)
        rate_decomp = 100.0 * e_target / n_target if n_target > 0 else last_rate

        # --- AR fallback: Theil-Sen slope of logit(rate), last 12h ---------
        xs12, ys12 = [], []
        for i in idx12:
            r = history[i]["rates"][j]
            if r is not None:
                xs12.append(times[i])
                ys12.append(logit(r))
        slope = theil_sen_slope(xs12, ys12) if len(xs12) >= 2 else 0.0
        slope = clamp(slope, -0.05, 0.05)
        logit_ar = logit(last_rate) + slope * h_eff

        # --- blend (decomposition primary, AR fallback) ----------------------
        # Sparse history (Superbot final review item #1): if the trophy's
        # earner count has taken fewer than ~3 distinct values, the
        # cumulative slope is mostly noise — raise the AR fallback weight
        # from 30% to 50%. Trophies with >=3 distinct E values keep 70/30.
        n_distinct_e = len({round(e_series[i][j], 3)
                            for i in range(len(history))
                            if e_series[i][j] is not None})
        if len(xs12) >= 3:
            w_ar = 0.5 if n_distinct_e < 3 else 0.3
        else:
            w_ar = 0.0
        logit_center = (1.0 - w_ar) * logit(rate_decomp) + w_ar * logit_ar
        center = inv_logit(logit_center)
        # Ceiling solve: history may override the prior, but a forecast
        # above the category ceiling is capped at it.
        if ceiling is not None:
            center = min(center, ceiling * 100.0)
        center = clamp(center, 0.01, 99.99)

        # --- classification (for the dilution scale) ------------------------
        if last_e is not None and last_e < 10:
            kind = "immature"
        elif slope > 0.005:
            kind = "absorbing"
        else:
            kind = "lagging"
        absorb_scale = {"absorbing": 0.5, "lagging": 1.0,
                        "immature": 0.75}[kind]

        # --- spread synthesis ------------------------------------------------
        # Residual spread: scatter of logit(rate) around its OLS trend,
        # last 24h (floor: the history may be perfectly flat).
        xs24, ys24 = [], []
        for i in idx24:
            r = history[i]["rates"][j]
            if r is not None:
                xs24.append(times[i])
                ys24.append(logit(r))
        if len(xs24) >= 3:
            b, a0 = ols_slope_intercept(xs24, ys24)
            res = [y - (a0 + b * x) for x, y in zip(xs24, ys24)]
            mres = sum(res) / len(res)
            var = sum((r - mres) ** 2 for r in res) / max(len(res) - 1, 1)
            sigma_resid = math.sqrt(var)
            # Standard error of the slope.
            sxx = sum((x - sum(xs24) / len(xs24)) ** 2 for x in xs24)
            se_slope = math.sqrt(var / sxx) if sxx > 0 else 0.0
        else:
            sigma_resid, se_slope = 0.0, 0.0
        sigma_resid = max(sigma_resid, 0.30)
        if tr.get("secret"):
            sigma_resid *= 1.15
        sigma_dilution = gamma * absorb_scale
        sigma = math.sqrt(sigma_resid ** 2 + (se_slope * h_eff) ** 2
                          + sigma_dilution ** 2)
        sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX)

        rows[tid] = quantiles(center, sigma)

    return rows


def quantiles(center_pct, sigma):
    lc = logit(center_pct)
    qs = [round4(clamp(inv_logit(lc + z * sigma), 0.01, 99.99)) for z in Z]
    # Monotone by construction (z increasing, sigma > 0); enforce anyway.
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        print("no .in files in %s" % INPUT_DIR, file=sys.stderr)
        return 1
    for fname in inputs:
        cid = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        if case.get("levels", LEVELS) != LEVELS:
            raise ValueError("%s asks for levels %r" % (fname, case.get("levels")))
        rows = forecast_case(case)
        out = {"trophies": [{"id": tid, "q": rows[tid]}
                            for tid in sorted(rows)]}
        text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("%s output %d bytes > 65536"
                             % (fname, len(text.encode("utf-8"))))
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
