"""Forecaster for gta6-trophy-rarity-forecast-01: earners-over-players, structurally extrapolated.

The case's history is the house's recording of Exophase's GTA VI PS5 trophy page: reads every few
minutes, each holding every trophy's rarity (percent of tracked players) and the players-tracked
count. The page only changes when Exophase refreshes its counts, so the informative object is the
sequence of distinct states, each a world state at its first read.

The forecast decomposes each trophy's earner count the way the page itself does:

    rate_i = 100 * E_i / P

with P the players tracked and E_i the implied earner count (rate x P / 100, a 2-decimal display,
so E_i carries that rounding). Earners only accumulate: E_i grows by new entrants who already hold
the trophy (a per-trophy share c_i of each tracked player added) and by tracked players earning it
while tracked (a per-trophy hazard h_i per player-hour, decaying as the launch cohort finishes the
content). A per-trophy least-squares over the state intervals estimates (c_i, h_i); the players
curve is extrapolated with decelerating families whose disagreement and a refit bootstrap carry the
uncertainty; samples draw P, rebuild E_i, and the quantiles are the sample percentiles of the rate.

When the page was frozen for the whole history (launch night, the practice), the states say
nothing about growth, so the players come from observed launch anchors and earners from the same
accumulation split by their share. Standard library only, deterministic (seeded by the case id).
"""

import hashlib
import math
import random

# Levels the contest scores, in output order.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Observed launch-night reads of the page (hours since 2026-11-19T00:00Z, players tracked): the
# launch probe (23:50Z slot), the first refresh the competition's notes publish (04:30Z read:
# 3,868 players, 4,276 trophies earned), and the first read this entry's build watched (15:56Z).
P_ANCHORS = (
    (0.0, 638.0),
    (4.5, 3868.0),
    (15.93, 14028.0),
)
# The launch-night frozen state's players count, and the refresh the anchors expect after it.
LAUNCH_FROZEN_PLAYERS = 638.0
LAUNCH_REFRESH_AFTER_HOURS = 2.8

# Sampling: samples per fit family; bootstrap refits per family.
SAMPLES_PER_FAMILY = 240
BOOTSTRAP_REPS = 20
POISSON_NORMAL_LAMBDA = 30.0

# The stale-page regime: at most this many distinct states, or the page holding one state for at
# least this share of the history's read span (and at least this many hours).
STALE_STATES_MAX = 3
STALE_GAP_FRACTION = 0.55
STALE_GAP_MIN_HOURS = 1.5

# The progression hazard's decay time constant, in days: the launch cohort finishes the content
# and stops re-earning; entrant carry (which rides the players curve) takes over.
TAU_H_DAYS = 3.0
# Entrant-carry share floor: a fresh entrant is unlikely to hold much beyond the automatic trophy.
CARRY_DAMP = 0.03

HOURS_PER_DAY = 24.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    c = clamp_pct(pct)
    return math.log(c / (100.0 - c))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_ts(s):
    """Hours since 2026-11-19T00:00:00Z for an ISO-8601 UTC timestamp."""
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    h = int(s[11:13]); mi = int(s[14:16]); se = int(s[17:19])
    days = _days_from_civil(y, mo, d) - _days_from_civil(2026, 11, 19)
    return days * 24.0 + h + mi / 60.0 + se / 3600.0


def _days_from_civil(y, m, d):
    """Days since 1970-01-01 (Howard Hinnant's algorithm)."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def distinct_states(history):
    """The page's states: consecutive reads collapsed while players and every rate stay equal."""
    states = []
    last_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == last_key:
            continue
        last_key = key
        states.append({
            "t": parse_ts(read.get("slot") or read.get("at")),
            "players": float(read.get("players_tracked") or 0),
            "rates": [float(r) if r is not None else 0.0 for r in rates],
        })
    return states


def auto_index(rates):
    for i, r in enumerate(rates):
        if r >= 99.0:
            return i
    return None


# ---------------------------------------------------------------------------
# The players curve: decelerating families on ln P, blended by fit quality.
# ---------------------------------------------------------------------------

def wls_fit(x, y, w):
    """Weighted least squares y ~ a + b x. Returns (a, b, sse, n) or None."""
    sw = sum(w)
    if sw <= 0 or len(x) < 3:
        return None
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    sxy = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y))
    if sxx <= 1e-12:
        return None
    b = sxy / sxx
    a = my - b * mx
    sse = sum(wi * (yi - (a + b * xi)) ** 2 for wi, xi, yi in zip(w, x, y))
    return (a, b, sse, len(x))


def quad_fit(x, y, w):
    """Weighted least squares y ~ a + b x + c x^2. Returns (a, b, c, sse, n) or None."""
    n = len(x)
    if n < 4:
        return None
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    mx2 = sum(wi * xi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    c11 = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    c12 = sum(wi * (xi - mx) * (xi * xi - mx2) for wi, xi in zip(w, x))
    c22 = sum(wi * (xi * xi - mx2) ** 2 for wi, xi in zip(w, x))
    r1 = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y))
    r2 = sum(wi * (xi * xi - mx2) * (yi - my) for wi, xi, yi in zip(w, x, y))
    det = c11 * c22 - c12 * c12
    if abs(det) < 1e-12 or c22 <= 1e-12:
        return None
    b = (r1 * c22 - c12 * r2) / det
    c = (c11 * r2 - c12 * r1) / det
    a = my - b * mx - c * mx2
    sse = sum(wi * (yi - (a + b * xi + c * xi * xi)) ** 2 for wi, xi, yi in zip(w, x, y))
    return (a, b, c, sse, n)


def p_families(t, y, w):
    """Fit the decelerating families to (t, ln P). Returns [(weight, fn, resid_sd), ...]."""
    n = len(t)
    x_log = [math.log(1.0 + max(ti, 0.0)) for ti in t]
    fits = []
    f1 = wls_fit(x_log, y, w)
    if f1:
        a, b, sse, k = f1
        fits.append((sse, k - 2, (lambda tt, a=a, b=b: a + b * math.log(1.0 + max(tt, 0.0)))))
    taus = (6.0, 9.0, 14.0, 22.0, 34.0, 50.0)
    best_sat = None
    for tau in taus:
        z = [1.0 - math.exp(-max(ti, 0.0) / tau) for ti in t]
        f = wls_fit(z, y, w)
        if f:
            a, b, sse, k = f
            if best_sat is None or sse < best_sat[0]:
                best_sat = (sse, k - 2, a, b, tau)
    if best_sat:
        sse, k, a, b, tau = best_sat
        fits.append((sse, k, (lambda tt, a=a, b=b, tau=tau: a + b * (1.0 - math.exp(-max(tt, 0.0) / tau)))))
    f3 = quad_fit(x_log, y, w)
    if f3:
        a, b, c, sse, k = f3
        fits.append((sse, k - 3, (lambda tt, a=a, b=b, c=c: a + b * math.log(1.0 + max(tt, 0.0)) + c * math.log(1.0 + max(tt, 0.0)) ** 2)))
    if not fits:
        return None
    scored = []
    for sse, k, fn in fits:
        bic = n * math.log(max(sse / n, 1e-12)) + (k + 1) * math.log(n)
        sd = math.sqrt(max(sse / max(n - max(k, 1), 1), 1e-9))
        scored.append((bic, fn, sd))
    lo = min(s[0] for s in scored)
    weights = [math.exp(-(s[0] - lo) / 2.0) for s in scored]
    total = sum(weights)
    return [(wt / total, fn, sd) for (bic, fn, sd), wt in zip(scored, weights)]


def p_spread(t, y, w, target_t, rng, reps=BOOTSTRAP_REPS):
    """Refit the families on residual bootstrap replicates; return the spread of the blend at T."""
    if len(t) < 4:
        return 0.0
    fams = p_families(t, y, w)
    if not fams:
        return 0.0
    # Each point's residual against its closest family, as the bootstrap's resampling pool.
    resid = []
    for ti, yi in zip(t, y):
        best = None
        for wt, fn, sd in fams:
            e = yi - fn(ti)
            if best is None or abs(e) < abs(best):
                best = e
        resid.append(best)
    preds = []
    for _ in range(reps):
        yb = [yi + rng.choice(resid) for yi in y]
        fb = p_families(t, yb, w)
        if not fb:
            continue
        preds.append(sum(wt * fn(target_t) for wt, fn, sd in fb))
    if len(preds) < 5:
        return 0.0
    mean = sum(preds) / len(preds)
    var = sum((p - mean) ** 2 for p in preds) / (len(preds) - 1)
    centers = [fn(target_t) for wt, fn, sd in fams]
    cmean = sum(wt * c for (wt, fn, sd), c in zip(fams, centers))
    cvar = sum(wt * (c - cmean) ** 2 for (wt, fn, sd), c in zip(fams, centers))
    return math.sqrt(var + cvar)


# ---------------------------------------------------------------------------
# The earner accumulation: dE_i = c_i dP + h_i P dt, estimated over state intervals.
# ---------------------------------------------------------------------------

def fit_carry_hazard(pts):
    """Least squares de ~ c*dp + h*pbar_dt over intervals, clipped to c in [0, 1], h >= 0."""
    if len(pts) < 2:
        return None
    s11 = sum(p[0] * p[0] for p in pts)
    s12 = sum(p[0] * p[1] for p in pts)
    s22 = sum(p[1] * p[1] for p in pts)
    s1y = sum(p[0] * p[2] for p in pts)
    s2y = sum(p[1] * p[2] for p in pts)
    det = s11 * s22 - s12 * s12
    c = h = 0.0
    if abs(det) > 1e-9:
        c = (s1y * s22 - s12 * s2y) / det
        h = (s11 * s2y - s12 * s1y) / det
    if abs(c - min(1.0, max(0.0, c))) > 1e-9 or abs(h - max(0.0, h)) > 1e-9:
        # Refit the free parameter with the other held at its bound.
        if c < 0:
            h = s2y / max(s22, 1e-9)
            c = 0.0
        elif c > 1:
            h = (s2y - s12) / max(s22, 1e-9)
            c = 1.0
        if h < 0:
            c = s1y / max(s11, 1e-9)
            h = 0.0
    return (min(1.0, max(0.0, c)), max(0.0, h))


def trophy_intervals(states, idx):
    """Per-state-interval (dp, pbar*dt, dE) rows for trophy idx."""
    pts = []
    for k in range(1, len(states)):
        a, b = states[k - 1], states[k]
        dt = b["t"] - a["t"]
        if dt <= 0.02:
            continue
        ea = a["rates"][idx] * a["players"] / 100.0
        eb = b["rates"][idx] * b["players"] / 100.0
        pts.append((b["players"] - a["players"], 0.5 * (a["players"] + b["players"]) * dt, eb - ea))
    return pts


def poisson_draw(rng, lam):
    if lam <= 0:
        return 0
    if lam >= POISSON_NORMAL_LAMBDA:
        return max(0, int(round(lam + math.sqrt(lam) * rng.gauss(0.0, 1.0))))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def quantiles_from(samples_row):
    col = sorted(samples_row)
    q = []
    for lv in LEVELS:
        idx = lv * (len(col) - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, len(col) - 1)
        frac = idx - lo
        q.append(round4(col[lo] * (1.0 - frac) + col[hi] * frac))
    return q


# ---------------------------------------------------------------------------
# The two regimes.
# ---------------------------------------------------------------------------

def forecast_rich(case, states, target_t):
    n_trophy = len(case["trophies"])
    t = [s["t"] for s in states]
    lnP = [math.log(max(s["players"], 1.0)) for s in states]
    w = [1.0] * len(t)
    fams = p_families(t, lnP, w)
    if not fams:
        return None
    rng = random.Random(int(hashlib.sha256(case["id"].encode()).hexdigest()[:16], 16))
    sd_P = p_spread(t, lnP, w, target_t, rng)
    horizon_days = max((target_t - t[-1]) / HOURS_PER_DAY, 0.02)
    sd_P = max(sd_P, 0.20 * math.sqrt(horizon_days))
    n_fam = len(fams)

    now = states[-1]
    now_t = now["t"]
    P_now = max(now["players"], 1.0)
    tau_h_hours = TAU_H_DAYS * HOURS_PER_DAY
    # The hazard's decay from now to the target, and the players-hour integral it weights, along
    # the blended players path.
    seg = 12
    path_t = [now_t + (target_t - now_t) * k / seg for k in range(seg + 1)]
    blend_at = lambda tt: sum(wt * fn(tt) for wt, fn, sd in fams)
    path_lnP = [blend_at(tt) for tt in path_t]
    decay_int = 0.0
    for k in range(seg):
        dt = path_t[k + 1] - path_t[k]
        p_mid = math.exp(0.5 * (path_lnP[k] + path_lnP[k + 1]))
        decay = math.exp(-((path_t[k] - now_t) + 0.5 * dt) / tau_h_hours)
        decay_int += p_mid * decay * dt
    dp_target = math.exp(blend_at(target_t)) - P_now

    # Each trophy's mean-count pool: the point forecast plus an interval bootstrap of (c, h),
    # which carries the estimation noise the display rounding and process leave behind.
    lam_pools = []
    for i in range(n_trophy):
        e_now = now["rates"][i] * P_now / 100.0
        pts = trophy_intervals(states, i)
        base = fit_carry_hazard(pts)
        pool = []
        if base is not None and len(pts) >= 3:
            c, h = base
            pool.append(e_now + c * dp_target + h * decay_int)
            for _ in range(BOOTSTRAP_REPS):
                resample = [pts[rng.randrange(len(pts))] for _ in range(len(pts))]
                ch = fit_carry_hazard(resample)
                if ch is None:
                    continue
                cb, hb = ch
                pool.append(e_now + cb * dp_target + hb * decay_int)
        else:
            share = min(now["rates"][i] / 100.0, 1.0)
            for _ in range(BOOTSTRAP_REPS):
                carry = share * (0.3 + 0.4 * rng.random())
                hz = max(0.0, e_now / max(P_now, 1.0) / HOURS_PER_DAY) * (0.5 + rng.random())
                pool.append(e_now + carry * dp_target + hz * decay_int * HOURS_PER_DAY)
        lam_pools.append([max(x, 0.05) for x in pool])

    auto = auto_index(now["rates"])
    samples = [[] for _ in range(n_trophy)]
    for m in range(n_fam):
        fn_P = fams[m][1]
        for _ in range(SAMPLES_PER_FAMILY):
            p_draw = math.exp(fn_P(target_t) + sd_P * rng.gauss(0.0, 1.0))
            if p_draw < P_now:
                p_draw = P_now
            for i in range(n_trophy):
                if i == auto:
                    samples[i].append(100.0)
                    continue
                e_now = now["rates"][i] * P_now / 100.0
                lam = lam_pools[i][rng.randrange(len(lam_pools[i]))]
                e_draw = poisson_draw(rng, lam)
                e_draw = max(e_draw, math.floor(e_now))
                rate = 100.0 * e_draw / p_draw
                if rate > 100.0:
                    rate = 100.0
                samples[i].append(rate)
    return [quantiles_from(samples[i]) for i in range(n_trophy)]


def forecast_stale(case, states, target_t):
    """The frozen-page path: the world grew while the page held one state."""
    n_trophy = len(case["trophies"])
    last = states[-1]
    t_last = last["t"]
    P_last = max(last["players"], 1.0)
    rates_last = last["rates"]
    E_last = [max(r * P_last / 100.0, 0.0) for r in rates_last]

    pts = sorted((tt, math.log(pp)) for tt, pp in P_ANCHORS)
    if abs(P_last - LAUNCH_FROZEN_PLAYERS) / LAUNCH_FROZEN_PLAYERS < 0.02:
        after = [p for p in pts if p[0] >= t_last + LAUNCH_REFRESH_AFTER_HOURS]
        if after and target_t >= after[0][0] - 0.5:
            mu_lnP = after[0][1]
            sd_P = 0.10
        else:
            mu_lnP = pts[0][1]
            sd_P = 0.45
    elif target_t <= pts[0][0]:
        mu_lnP = pts[0][1]
        sd_P = 0.5
    elif target_t >= pts[-1][0]:
        (t1, y1), (t2, y2) = pts[-2], pts[-1]
        slope = (y2 - y1) / max(t2 - t1, 0.25)
        gap = target_t - t2
        damp = math.exp(-gap / (TAU_H_DAYS * HOURS_PER_DAY))
        mu_lnP = y2 + slope * (TAU_H_DAYS * HOURS_PER_DAY) * (1.0 - damp)
        sd_P = 0.35 + 0.08 * math.sqrt(max(gap, 1.0))
    else:
        mu_lnP = pts[0][1]
        sd_P = 0.3
        for k in range(1, len(pts)):
            if target_t <= pts[k][0]:
                (t1, y1), (t2, y2) = pts[k - 1], pts[k]
                f = (target_t - t1) / max(t2 - t1, 1e-9)
                mu_lnP = y1 + f * (y2 - y1)
                sd_P = 0.18 + 0.06 * (t2 - t1) / 4.0
                break
    rng = random.Random(int(hashlib.sha256((case["id"] + "stale").encode()).hexdigest()[:16], 16))

    P_target = math.exp(mu_lnP)
    new_players = max(P_target - P_last, 0.0)
    auto = auto_index(rates_last)
    share_sum = 0.0
    for r in rates_last:
        if r < 99.0 and r > 0:
            share_sum += r
    prog_total = max(target_t - max(t_last, 0.0), 0.5) * 22.0 * (P_last / 638.0) ** 0.35
    E_mu = []
    for i in range(n_trophy):
        e = E_last[i]
        if i == auto:
            e = e + new_players
        else:
            carry = new_players * (E_last[i] / max(P_last, 1.0)) * CARRY_DAMP
            share = (rates_last[i] / share_sum) if share_sum > 0 and rates_last[i] > 0 else 0.0
            e = e + carry + prog_total * share
        E_mu.append(max(e, 0.15))

    samples = [[] for _ in range(n_trophy)]
    for _ in range(SAMPLES_PER_FAMILY * 4):
        p_draw = math.exp(mu_lnP + sd_P * rng.gauss(0.0, 1.0))
        if p_draw < 1.0:
            p_draw = 1.0
        for i in range(n_trophy):
            if i == auto:
                samples[i].append(100.0)
                continue
            lam = E_mu[i] * math.exp(0.40 * rng.gauss(0.0, 1.0))
            e_draw = poisson_draw(rng, lam)
            rate = 100.0 * e_draw / p_draw
            if rate > 100.0:
                rate = 100.0
            samples[i].append(rate)
    return [quantiles_from(samples[i]) for i in range(n_trophy)]


def is_stale(states, first_t, last_t):
    if len(states) <= STALE_STATES_MAX:
        return True
    gap = last_t - states[-1]["t"]
    span = max(last_t - first_t, 1e-9)
    if gap > max(STALE_GAP_MIN_HOURS, STALE_GAP_FRACTION * span):
        return True
    return False


def forecast_case(case):
    history = case.get("history") or []
    if not history:
        raise ValueError("no history")
    levels = case.get("levels")
    if list(levels) != list(LEVELS):
        raise ValueError("unexpected levels")
    target_t = parse_ts(case["target"])
    states = distinct_states(history)
    if not states:
        raise ValueError("no states")
    first_t = parse_ts(history[0].get("slot") or history[0].get("at"))
    last_t = parse_ts(history[-1].get("slot") or history[-1].get("at"))
    if is_stale(states, first_t, last_t):
        q = forecast_stale(case, states, target_t)
    else:
        q = forecast_rich(case, states, target_t)
        if q is None:
            q = forecast_stale(case, states, target_t)
    rows = []
    for trophy, qi in zip(case["trophies"], q):
        rows.append({"id": trophy["id"], "q": qi})
    return {"trophies": rows}
