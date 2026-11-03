import numpy as np


def _vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("A finite nonempty vector is required")
    return values


def planning_distance(actions, reference):
    actions, reference = _vector(actions), _vector(reference)
    if actions.shape != reference.shape:
        raise ValueError("Trajectory lengths differ")
    return float(np.abs(actions - reference).sum())


def variance_ratio(orders, demand):
    orders, demand = _vector(orders), _vector(demand)
    if orders.shape != demand.shape or len(demand) < 2:
        raise ValueError("Variance ratio requires paired trajectories of length at least two")
    variance = float(demand.var(ddof=1))
    return float(orders.var(ddof=1) / variance) if variance else None


def local_chasing(orders, demand):
    orders, demand = _vector(orders), _vector(demand)
    if orders.shape != demand.shape or len(demand) < 3:
        raise ValueError("Lag-one correlation requires paired trajectories of length at least three")
    x, y = orders[1:], demand[:-1]
    return float(np.corrcoef(x, y)[0, 1]) if x.var() and y.var() else None


def fulfillment_statistics(demand, immediately_fulfilled):
    demand, fulfilled = _vector(demand), _vector(immediately_fulfilled)
    if demand.shape != fulfilled.shape or (demand < 0).any() or (fulfilled < 0).any() or (fulfilled > demand).any():
        raise ValueError("Invalid immediate-fulfillment records")
    positive = demand > 0
    return {"success": float(np.mean(fulfilled[positive] == demand[positive])) if positive.any() else None,
        "service": float(fulfilled.sum() / demand.sum()) if demand.sum() else None}


def normalized_chain_cost(holding, backlog, nodes=4, holding_rate=1.0, backlog_rate=2.0):
    holding, backlog = _vector(holding), _vector(backlog)
    if type(nodes) is not int or nodes < 1 or min(holding_rate, backlog_rate) < 0 or not np.isfinite([holding_rate, backlog_rate]).all() or (holding < 0).any() or (backlog < 0).any():
        raise ValueError("Invalid cost inputs")
    return float((holding_rate * holding.sum() + backlog_rate * backlog.sum()) / nodes)
