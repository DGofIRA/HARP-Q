from collections import Counter
from itertools import product
from math import isfinite


def assignments(rows, capacities=None, roles=("Q1", "Q2", "Q3")):
    rows, roles = list(rows), tuple(roles)
    ids = [r["candidate"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or not roles or len(set(roles)) != len(roles):
        raise ValueError("Candidate identifiers and role names must be unique")
    if any(not isfinite(float(r[role])) for r in rows for role in roles):
        raise ValueError("Missing or nonfinite qualification")
    caps = dict.fromkeys(ids, 1)
    if capacities is not None:
        if set(capacities) - set(ids):
            raise ValueError("Capacity supplied for an unknown candidate")
        caps.update(capacities)
    if any(type(c) is not int or c < 0 for c in caps.values()):
        raise ValueError("Capacity must be a nonnegative integer")
    result = []
    for indices in product(range(len(rows)), repeat=len(roles)):
        chosen = [ids[i] for i in indices]
        if any(count > caps[candidate] for candidate, count in Counter(chosen).items()):
            continue
        objective = sum(rows[i][role] for i, role in zip(indices, roles))
        result.append({"assignment": chosen, "objective": objective})
    return sorted(result, key=lambda r: (-r["objective"], r["assignment"]))


def assignment_summary(rows, capacities=None, roles=("Q1", "Q2", "Q3"), near=0.02, tolerance=1e-12):
    if not isfinite(near) or near < 0 or not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid score-distance tolerance")
    results = assignments(rows, capacities, roles)
    if not results:
        raise ValueError("No feasible assignment")
    best = results[0]["objective"]
    return {"roles": list(roles), "objective": best,
        "optima": [r["assignment"] for r in results if best - r["objective"] <= tolerance],
        "best_second_gap": max(0.0, best - results[1]["objective"]) if len(results) > 1 else None,
        "near_tolerance": near,
        "near_optimal_count": sum(best - r["objective"] <= near + tolerance for r in results),
        "feasible_assignments": len(results)}
