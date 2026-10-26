def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default


def poisson_sample(rng, lam: float) -> int:
    """Sample from a Poisson distribution using numpy rng."""
    return int(rng.poisson(lam))
