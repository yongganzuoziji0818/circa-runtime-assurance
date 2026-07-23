"""Compiler and validator for the frozen CIRCA exact-truth fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .circa import INEVITABLE_VIOLATION, NOMINAL_SAFETY, NO_WITNESS, OBSERVED_CONSISTENCY
except ImportError:  # Direct, dependency-isolated G0 loading.
    from circa import INEVITABLE_VIOLATION, NOMINAL_SAFETY, NO_WITNESS, OBSERVED_CONSISTENCY


WITNESS_CODES = {
    "none": NO_WITNESS,
    "observed_consistency": OBSERVED_CONSISTENCY,
    "INEVITABLE_VIOLATION_WITNESS": INEVITABLE_VIOLATION,
    "NOMINAL_SAFETY_WITNESS": NOMINAL_SAFETY,
}


@dataclass(frozen=True)
class AtomicFamily:
    family_id: str
    probabilities: np.ndarray
    r: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    witness: np.ndarray
    pi0: np.ndarray
    m0: np.ndarray
    truth: dict


def load_fixture(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    if fixture.get("status") != "frozen_exact_truth_fixture_contract":
        raise ValueError("fixture contract is not frozen")
    if len(fixture.get("families", [])) != 6:
        raise ValueError("exactly six fixture families are required")
    return fixture


def _compile_f1(family: dict) -> AtomicFamily:
    model = family["model"]
    p, r, y0, y1, witness, pi0, m0 = [], [], [], [], [], [], []
    q = float(model["P_Y1_1_given_R1_Y0_1"])
    for z in ("low", "high"):
        pz = float(model["P_Z"][z])
        pr1 = float(model["P_R1_given_Z"][z])
        py0 = float(model["P_Y0_1_given_Z"][z])
        for rv in (0, 1):
            pr = pr1 if rv else 1.0 - pr1
            for y0v in (0, 1):
                py = py0 if y0v else 1.0 - py0
                outcomes = [(y0v, 1.0)] if rv == 0 else ([(0, 1.0)] if y0v == 0 else [(0, 1.0 - q), (1, q)])
                for y1v, py1 in outcomes:
                    p.append(pz * pr * py * py1)
                    r.append(rv)
                    y0.append(y0v)
                    y1.append(y1v)
                    witness.append(OBSERVED_CONSISTENCY if rv == 0 else NO_WITNESS)
                    pi0.append(1.0 - pr1)
                    m0.append(py0)
    return _family(family, p, r, y0, y1, witness, pi0, m0)


def _family(family: dict, p, r, y0, y1, witness, pi0, m0) -> AtomicFamily:
    result = AtomicFamily(
        family_id=family["id"],
        probabilities=np.asarray(p, dtype=float),
        r=np.asarray(r, dtype=np.int8),
        y0=np.asarray(y0, dtype=np.int8),
        y1=np.asarray(y1, dtype=np.int8),
        witness=np.asarray(witness, dtype=np.int8),
        pi0=np.asarray(pi0, dtype=float),
        m0=np.asarray(m0, dtype=float),
        truth=family["truth"],
    )
    _validate_atomic(result)
    return result


def _compile_atoms(family: dict) -> AtomicFamily:
    atoms = family["atoms"]
    return _family(
        family,
        [a["p"] for a in atoms],
        [a["R"] for a in atoms],
        [a["Y0"] for a in atoms],
        [a["Y1"] for a in atoms],
        [WITNESS_CODES[a["witness"]] for a in atoms],
        [np.nan] * len(atoms),
        [np.nan] * len(atoms),
    )


def _validate_atomic(family: AtomicFamily) -> None:
    if not np.isclose(family.probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError(f"{family.family_id}: probabilities do not sum to one")
    if np.any(family.probabilities < 0):
        raise ValueError(f"{family.family_id}: negative probability")
    p = family.probabilities
    ey0 = float(p @ family.y0)
    ey1 = float(p @ family.y1)
    intervention = float(p @ family.r)
    l_m = (1 - family.r) * family.y1
    u_m = l_m + family.r
    lower = l_m.astype(float)
    upper = u_m.astype(float)
    inevitable = (family.r == 1) & (family.witness == INEVITABLE_VIOLATION)
    nominal = (family.r == 1) & (family.witness == NOMINAL_SAFETY)
    lower[inevitable] = upper[inevitable] = 1.0
    lower[nominal] = upper[nominal] = 0.0
    expected = family.truth
    checks = {
        "E_Y0": ey0,
        "E_Y1": ey1,
        "delta": ey0 - ey1,
        "intervention_rate": intervention,
    }
    for key, value in checks.items():
        if not np.isclose(value, expected[key], atol=1e-12):
            raise ValueError(f"{family.family_id}: {key} mismatch {value} != {expected[key]}")
    manski = (float(p @ (l_m - family.y1)), float(p @ (u_m - family.y1)))
    circa = (float(p @ (lower - family.y1)), float(p @ (upper - family.y1)))
    if not np.allclose(manski, expected["manski_interval"], atol=1e-12):
        raise ValueError(f"{family.family_id}: Manski truth mismatch")
    if not np.allclose(circa, expected["circa_interval"], atol=1e-12):
        raise ValueError(f"{family.family_id}: CIRCA truth mismatch")
    ratio = (circa[1] - circa[0]) / (manski[1] - manski[0])
    if not np.isclose(ratio, expected["width_ratio"], atol=1e-12):
        raise ValueError(f"{family.family_id}: width ratio mismatch")
    if np.any(lower > family.y0) or np.any(upper < family.y0):
        raise ValueError(f"{family.family_id}: structured bound is unsound")
    if np.any(l_m > lower) or np.any(upper > u_m):
        raise ValueError(f"{family.family_id}: structured bound is not nested")


def compile_atomic_families(fixture: dict) -> dict[str, AtomicFamily]:
    families = fixture["families"]
    compiled = {families[0]["id"]: _compile_f1(families[0])}
    for family in families[1:4]:
        compiled[family["id"]] = _compile_atoms(family)
    if float(families[0]["truth"]["delta"]) > float(fixture["delta_star"]):
        raise ValueError("F1 is not a valid null family")
    return compiled
