from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FormulationBounds:
    """
    Bounds for the formulation + process variables.

    Composition variables are in wt% and should sum to ~100 (see generator).
    """

    polymer_A: Tuple[float, float] = (40.0, 70.0)
    polymer_B: Tuple[float, float] = (10.0, 30.0)
    plasticizer: Tuple[float, float] = (0.0, 20.0)
    filler: Tuple[float, float] = (0.0, 25.0)
    stabilizer: Tuple[float, float] = (0.0, 5.0)

    temperature: Tuple[float, float] = (150.0, 250.0)  # °C
    curing_time: Tuple[float, float] = (5.0, 60.0)  # min
    pressure: Tuple[float, float] = (1.0, 10.0)  # bar


def generate_synthetic_formulations(
    n_samples: int,
    *,
    seed: Optional[int] = 0,
    bounds: FormulationBounds = FormulationBounds(),
    composition_sum: float = 100.0,
    sum_tolerance: float = 0.25,
    noise_level: float = 1.0,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Generate a realistic synthetic dataset for polymer cable coating formulation optimization.

    What this function enforces:
    - Composition variables respect realistic bounds.
    - Composition sums to ~100 wt% (within sum_tolerance).
    - Process variables are within realistic bounds.
    - Targets follow physically-motivated relationships with non-linearities + controlled noise.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    seed:
        Random seed for reproducibility.
    bounds:
        Bounds container for all variables.
    composition_sum:
        Desired sum for composition variables (wt%).
    sum_tolerance:
        Allowed absolute deviation for composition sum, e.g. 0.25 means 99.75–100.25.
    noise_level:
        Global multiplier for output noise (1.0 is default). Set lower for cleaner targets.
    return_metadata:
        If True, return (df, metadata) including generation settings.

    Returns
    -------
    pandas.DataFrame (or (DataFrame, metadata))
        Columns: inputs + targets.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if composition_sum <= 0:
        raise ValueError("composition_sum must be positive.")
    if sum_tolerance < 0:
        raise ValueError("sum_tolerance must be >= 0.")
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0.")

    rng = np.random.default_rng(seed)

    comp_names = ["polymer_A", "polymer_B", "plasticizer", "filler", "stabilizer"]
    comp_bounds = np.array([getattr(bounds, n) for n in comp_names], dtype=float)  # shape (5, 2)
    lo = comp_bounds[:, 0]
    hi = comp_bounds[:, 1]

    compositions = _sample_compositions_with_sum(
        rng=rng,
        n_samples=n_samples,
        lo=lo,
        hi=hi,
        total=composition_sum,
        tol=sum_tolerance,
        max_iter=50_000,
    )
    comp_df = pd.DataFrame(compositions, columns=comp_names)
    comp_df["composition_sum"] = comp_df[comp_names].sum(axis=1)

    # Process variables
    temperature = rng.uniform(bounds.temperature[0], bounds.temperature[1], size=n_samples)
    curing_time = rng.uniform(bounds.curing_time[0], bounds.curing_time[1], size=n_samples)
    pressure = rng.uniform(bounds.pressure[0], bounds.pressure[1], size=n_samples)

    X = comp_df.copy()
    X["temperature"] = temperature
    X["curing_time"] = curing_time
    X["pressure"] = pressure

    # Targets (physically-motivated surrogate truth)
    y = _compute_targets(X, rng=rng, noise_level=noise_level)

    df = pd.concat([X, y], axis=1)

    if return_metadata:
        meta = {
            "seed": seed,
            "n_samples": n_samples,
            "bounds": bounds,
            "composition_sum": composition_sum,
            "sum_tolerance": sum_tolerance,
            "noise_level": noise_level,
        }
        return df, meta
    return df


def simulate_lab_results(
    X: pd.DataFrame,
    *,
    seed: Optional[int] = 0,
    noise_level: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate "lab measurements" for candidate formulations.

    This is the key building block for an iterative optimization loop:
    - propose next candidate(s)
    - run (simulated) experiment to get measured targets
    - append to dataset and retrain

    In an actual experimental campaign this function would be replaced by real laboratory measurements.

    Parameters
    ----------
    X:
        DataFrame containing the input variables used by the generator:
        composition columns + process variables.
    seed:
        Random seed to make simulated measurements reproducible.
    noise_level:
        Noise multiplier (1.0 default). Use >1 to simulate noisier labs.

    Returns
    -------
    pandas.DataFrame
        Targets: tensile_strength, elongation, thermal_resistance.
    """

    if noise_level < 0:
        raise ValueError("noise_level must be >= 0.")
    rng = np.random.default_rng(seed)
    return _compute_targets(X, rng=rng, noise_level=noise_level)


def _sample_compositions_with_sum(
    *,
    rng: np.random.Generator,
    n_samples: int,
    lo: np.ndarray,
    hi: np.ndarray,
    total: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Sample compositions x in [lo, hi] such that sum(x) ~= total.

    Strategy:
    - Sample uniformly within bounds.
    - Project to the simplex by scaling to the desired total.
    - Reject if any variable violates bounds after scaling.

    This is efficient for moderate dimensions (here 5 vars) and produces
    diverse points while keeping bounds realistic.
    """

    d = lo.shape[0]
    if d != hi.shape[0]:
        raise ValueError("lo and hi must have same length.")
    if np.any(hi <= lo):
        raise ValueError("Each upper bound must be > lower bound.")

    out = np.empty((n_samples, d), dtype=float)
    filled = 0
    attempts = 0

    # Oversample per batch to reduce Python overhead.
    batch = max(512, min(8192, n_samples * 8))

    while filled < n_samples and attempts < max_iter:
        attempts += 1
        # Uniform in hyper-rectangle
        x = rng.uniform(lo, hi, size=(batch, d))
        s = x.sum(axis=1, keepdims=True)
        # Avoid division by zero (shouldn't happen given bounds)
        s = np.clip(s, 1e-9, None)
        x = x * (total / s)

        # Accept only if within bounds and sum within tolerance.
        within_bounds = np.all((x >= lo[None, :]) & (x <= hi[None, :]), axis=1)
        within_sum = np.abs(x.sum(axis=1) - total) <= tol
        ok = within_bounds & within_sum
        if not np.any(ok):
            continue

        accepted = x[ok]
        take = min(accepted.shape[0], n_samples - filled)
        out[filled : filled + take] = accepted[:take]
        filled += take

    if filled < n_samples:
        raise RuntimeError(
            "Could not sample enough valid compositions. "
            "Try increasing sum_tolerance, widening bounds, or increasing max_iter."
        )

    return out


def _compute_targets(
    X: pd.DataFrame, *, rng: np.random.Generator, noise_level: float
) -> pd.DataFrame:
    """
    Synthetic 'ground truth' that encodes domain intuition + non-linearities.

    Targets:
    - tensile_strength (MPa): increases with filler; decreases with plasticizer; has an
      optimum temperature band; mild benefits from curing/pressure with diminishing returns.
    - elongation (%): increases with plasticizer; decreases with filler; penalized when
      temperature is too low/high; slightly reduced by high stabilizer (stiffer matrix).
    - thermal_resistance (°C): increases with stabilizer; has a mild temperature optimum;
      can be hurt by too much plasticizer (volatilization/softening).
    """

    pa = X["polymer_A"].to_numpy()
    pb = X["polymer_B"].to_numpy()
    pl = X["plasticizer"].to_numpy()
    fi = X["filler"].to_numpy()
    st = X["stabilizer"].to_numpy()

    T = X["temperature"].to_numpy()
    t = X["curing_time"].to_numpy()
    p = X["pressure"].to_numpy()

    # Temperature optimality: bell-shaped response around ~205°C with width ~18°C.
    # Normalize for stable magnitude across outputs.
    temp_opt_center = 205.0
    temp_opt_width = 18.0
    temp_opt = np.exp(-0.5 * ((T - temp_opt_center) / temp_opt_width) ** 2)  # in (0, 1]

    # Diminishing returns (0..1) for curing/pressure.
    cure_effect = 1.0 - np.exp(-(t - 5.0) / 18.0)
    cure_effect = np.clip(cure_effect, 0.0, 1.0)
    press_effect = 1.0 - np.exp(-(p - 1.0) / 3.5)
    press_effect = np.clip(press_effect, 0.0, 1.0)

    # Interaction: filler benefits depend on plasticizer (too much plasticizer weakens network).
    filler_gain = fi * (1.0 - 0.015 * pl)

    # Tensile strength (MPa)
    strength = (
        18.0
        + 0.55 * filler_gain
        - 0.35 * pl
        + 0.06 * (pa - pb)  # slight dependence on base polymer mix
        + 8.5 * temp_opt
        + 2.8 * cure_effect
        + 1.8 * press_effect
        - 0.015 * (fi**2)  # overfilling hurts strength
    )

    # Elongation (%)
    elong = (
        120.0
        + 4.2 * pl
        - 3.2 * fi
        + 18.0 * temp_opt
        + 8.0 * cure_effect
        - 1.2 * st
        - 0.06 * (pl * fi)  # classic trade-off interaction
    )

    # Thermal resistance (°C)
    therm = (
        85.0
        + 6.5 * st
        + 6.0 * temp_opt
        + 2.0 * cure_effect
        + 0.8 * press_effect
        - 0.25 * pl
        + 0.04 * fi  # small filler contribution
    )

    # Heteroscedastic-ish noise (more filler -> more variability).
    sigma_strength = noise_level * (0.8 + 0.03 * fi)
    sigma_elong = noise_level * (3.0 + 0.10 * pl + 0.08 * fi)
    sigma_therm = noise_level * (0.9 + 0.05 * st)

    strength = strength + rng.normal(0.0, sigma_strength, size=strength.shape[0])
    elong = elong + rng.normal(0.0, sigma_elong, size=elong.shape[0])
    therm = therm + rng.normal(0.0, sigma_therm, size=therm.shape[0])

    # Keep outputs in plausible ranges.
    strength = np.clip(strength, 5.0, 80.0)
    elong = np.clip(elong, 20.0, 450.0)
    therm = np.clip(therm, 40.0, 200.0)

    return pd.DataFrame(
        {
            "tensile_strength": strength,
            "elongation": elong,
            "thermal_resistance": therm,
        }
    )
