from itertools import combinations
from math import sqrt
from .scoring import score_panel, qualifications
from .assignment import assignment_summary


def _rank_comparison(first, second, role):
    a = {r["candidate"]: r[role] for r in first}
    b = {r["candidate"]: r[role] for r in second}
    c = d = ta = tb = 0
    for i, j in combinations(sorted(b), 2):
        x, y = a[i] - a[j], b[i] - b[j]
        if abs(x) < 1e-12 and abs(y) < 1e-12:
            continue
        if abs(x) < 1e-12:
            ta += 1
        elif abs(y) < 1e-12:
            tb += 1
        elif x * y > 0:
            c += 1
        else:
            d += 1
    denominator = sqrt((c + d + ta) * (c + d + tb))
    return {"tau_b": (c - d) / denominator if denominator else None, "reversals": d}


def _case(scores):
    return {**assignment_summary(scores), "zero_q": sum(r[q] == 0 for r in scores for q in ("Q1", "Q2", "Q3")), "scores": scores}


def diagnostics(rows):
    rows = list(rows)
    primary = score_panel(rows)
    normalization = {name: _case(score_panel(rows, normalization=name)) for name in ("endpoint", "midpoint", "minmax", "compressed_minmax")}
    loo = {}
    for row in rows:
        candidate = row["candidate"]
        retained = [r for r in rows if r["candidate"] != candidate]
        scored = score_panel(retained)
        fixed = [r for r in primary if r["candidate"] != candidate]
        loo[candidate] = {**_case(scored), "rank_comparison": {q: _rank_comparison(fixed, scored, q) for q in ("Q1", "Q2", "Q3")}, "restricted_full_panel": assignment_summary(fixed)}
    gates = {f"weight_{w:.2f}": _case(qualifications(primary, weight=w)) for w in (0.5, 0.25, 0.75)}
    gates.update({name: _case(qualifications(primary, gate=name)) for name in ("min", "sqrt", "none", "arithmetic")})
    overlaps = {}
    for name, options in {"primary": {}, "without_mpr_in_g": {"exclude_mpr": True}, "distance_only_p": {"distance_only": True}, "both_exclusions": {"exclude_mpr": True, "distance_only": True}, "turnover_as_burden": {"turnover": True}}.items():
        scored = score_panel(rows, **options)
        overlaps[name] = {**_case(scored), "maximum_abs_delta_q": max(abs(a[q] - b[q]) for a, b in zip(primary, scored) for q in ("Q1", "Q2", "Q3"))}
    return {"normalization": normalization, "leave_one_out": loo, "gates": gates, "overlap": overlaps}
