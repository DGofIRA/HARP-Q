from math import isfinite, log, sqrt


def _finite(values):
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("A finite numeric metric vector is required") from exc
    if not values or not all(isfinite(v) for v in values):
        raise ValueError("A nonempty finite metric vector is required")
    return values


def normalize(values, method="midpoint"):
    values = _finite(values)
    if method not in {"midpoint", "endpoint", "minmax", "compressed_minmax"}:
        raise ValueError("Unknown normalization")
    n = len(values)
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0.5] * n
    if method in {"minmax", "compressed_minmax"}:
        scaled = [(v - lo) / (hi - lo) for v in values]
        return scaled if method == "minmax" else [0.5 / n + (1 - 1 / n) * v for v in scaled]
    ranks = [1 + sum(x < v for x in values) + (sum(x == v for x in values) - 1) / 2 for v in values]
    return [(r - 0.5) / n for r in ranks] if method == "midpoint" else [(r - 1) / (n - 1) for r in ranks]


def harmonic(values):
    values = _finite(values)
    if min(values) < 0:
        raise ValueError("Harmonic inputs must be nonnegative")
    return 0.0 if min(values) == 0 else len(values) / sum(1 / v for v in values)


def weighted_harmonic(a, b, weight=0.5):
    a, b, weight = _finite([a, b, weight])
    if min(a, b) < 0 or not 0 <= weight <= 1:
        raise ValueError("Invalid weighted harmonic inputs")
    if weight == 0:
        return b
    if weight == 1:
        return a
    return 0.0 if min(a, b) == 0 else 1 / (weight / a + (1 - weight) / b)


def capabilities(rows, normalization="midpoint", exclude_mpr=False, distance_only=False, turnover=False):
    rows = list(rows)
    ids = [row["candidate"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Candidate identifiers must be nonempty and unique")
    for row in rows:
        for env in ("MPR", "TWN", "SCN"):
            cost, success = _finite([row["operating"][env]["cost"], row["operating"][env]["success"]])
            if cost < 0 or not 0 <= success <= 1:
                raise ValueError("Invalid operating summary")
        if float(row["planning_distance"]) < 0 or float(row["bg"]["beta"]) <= 0 or not -1 <= float(row["bg"]["rho"]) <= 1:
            raise ValueError("Invalid planning or BG summary")

    def phi(fn):
        return normalize([fn(row) for row in rows], normalization)

    operating = {}
    for env in ("MPR", "TWN", "SCN"):
        cols = [phi(lambda r: -r["operating"][env]["cost"]), phi(lambda r: r["operating"][env]["success"])]
        if turnover:
            cols.append(phi(lambda r: -r["operating"][env]["turnover_field"]))
        operating[env] = [harmonic(v) for v in zip(*cols)]
    ap = [(abs(r["nvp"]["pf_after"]) + abs(r["nvp"]["nf_after"])) / 2 for r in rows]
    pre = [(abs(r["nvp"]["pf_before"]) + abs(r["nvp"]["nf_before"])) / 2 for r in rows]
    fp = [abs(r["nvp"]["pf_after"] - r["nvp"]["nf_after"]) for r in rows]
    uc = [normalize([-v for v in ap], normalization), normalize([-v for v in fp], normalization)]
    pc = [phi(lambda r: -r["planning_distance"])]
    if not distance_only:
        pc.append(phi(lambda r: -r["operating"]["MPR"]["cost"]))
    sc = [phi(lambda r: -max(r["bg"]["rho"], 0)), phi(lambda r: -abs(log(r["bg"]["beta"])))]
    envs = ("TWN", "SCN") if exclude_mpr else ("MPR", "TWN", "SCN")
    result = []
    for i, candidate in enumerate(ids):
        result.append({"candidate": candidate,
            "G": harmonic([operating[e][i] for e in envs]),
            "U": harmonic([v[i] for v in uc]), "P": harmonic([v[i] for v in pc]),
            "S": harmonic([v[i] for v in sc]), "operating_scores": {e: operating[e][i] for e in operating},
            "A_pre": pre[i], "A_post": ap[i], "F_post": fp[i], "reflection_change": pre[i] - ap[i]})
    return result


def qualifications(rows, gate="product", weight=0.5):
    if gate not in {"product", "min", "sqrt", "none", "arithmetic"}:
        raise ValueError("Unknown gate rule")
    if not isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Weight must lie in [0, 1]")
    result = []
    for row in rows:
        g, u, p, s = _finite([row[k] for k in ("G", "U", "P", "S")])
        if not all(0 <= v <= 1 for v in (g, u, p, s)):
            raise ValueError("Capabilities must lie in [0, 1]")
        psi = g * p * s
        h1, h2 = harmonic([g, u]), harmonic([g, p, s])
        if gate == "none":
            q = (h1, h2, h2)
        elif gate == "arithmetic":
            q = ((u + (g + u) / 2) / 2, (s + (g + p + s) / 3) / 2, (psi + (g + p + s) / 3) / 2)
        else:
            penalty = {"product": psi, "min": min(g, p, s), "sqrt": sqrt(psi)}[gate]
            q = (weighted_harmonic(u, h1, weight), weighted_harmonic(s, h2, weight), weighted_harmonic(penalty, h2, weight))
        result.append({**row, "Psi": psi, **dict(zip(("Q1", "Q2", "Q3"), q))})
    return result


def score_panel(rows, normalization="midpoint", gate="product", weight=0.5, **options):
    return qualifications(capabilities(rows, normalization, **options), gate, weight)
