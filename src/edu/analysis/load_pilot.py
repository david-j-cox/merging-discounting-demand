"""Load DataPipe / Phase 4 JSON payloads into analysis-ready matrices.

Reads session JSONs (schema in ``experiment/src/lib/dataExport.ts``),
validates the fields the Bayesian fitter needs, and stacks them into
``(P, Q_obs, E, SV_obs, B_anchor, arm)``. Unknown extra fields are
preserved on ``LoadedSession.raw`` for forward compatibility;
validation failures raise :class:`InvalidPayload` with the source path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


class InvalidPayload(ValueError):
    """Raised when a session JSON does not match the expected schema."""


@dataclass(frozen=True)
class LoadedSession:
    """Per-subject data assembled from one session JSON."""

    session_id: str
    arm: str  # "low" or "high"
    p_max: float
    P: FloatArray  # purchase prices
    Q_obs: FloatArray  # consumption per price
    E: FloatArray  # effort levels (in p_max units, i.e. fraction * p_max)
    SV_obs: FloatArray  # subjective value per effort level
    raw: dict[str, Any]  # full original payload, for QA


@dataclass(frozen=True)
class LoadedCohort:
    """Stacked cohort matrices for the Bayesian fitter.

    Shapes: ``P`` ``(n_prices,)``; ``Q_obs`` ``(n_subj, n_prices)``;
    ``E``, ``SV_obs`` ``(n_subj, n_effort)``; ``B_anchor`` ``(n_subj,)``;
    ``arm`` ``(n_subj,)`` of ``"low"``/``"high"``. The price array must
    be identical across sessions (assembly raises otherwise).
    """

    sessions: list[LoadedSession]
    P: FloatArray
    Q_obs: FloatArray
    E: FloatArray
    SV_obs: FloatArray
    B_anchor: FloatArray
    arm: NDArray[np.str_]


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        msg = f"{where}: missing required field {key!r}"
        raise InvalidPayload(msg)
    return d[key]


def parse_session(payload: dict[str, Any], *, source: str = "<dict>") -> LoadedSession:
    """Convert one ``SessionPayload`` dict into a ``LoadedSession``.

    Validates the fields the Bayesian fitter needs. Unknown extra fields
    are kept on ``LoadedSession.raw`` for QA and not validated.
    """
    session_id = str(_require(payload, "sessionId", source))
    randomization = _require(payload, "randomization", source)
    arm = str(_require(randomization, "arm", f"{source}.randomization"))
    if arm not in ("low", "high"):
        msg = f"{source}: arm must be 'low' or 'high'; got {arm!r}"
        raise InvalidPayload(msg)

    calibration = _require(payload, "calibration", source)
    p_max_raw = _require(calibration, "pMax", f"{source}.calibration")
    p_max = float(p_max_raw)
    if not np.isfinite(p_max) or p_max <= 0:
        msg = f"{source}: pMax must be a positive finite number; got {p_max_raw!r}"
        raise InvalidPayload(msg)

    task1 = _require(payload, "task1", source)
    trials = _require(task1, "trials", f"{source}.task1")
    if not isinstance(trials, list) or len(trials) == 0:
        msg = f"{source}.task1.trials must be a non-empty list"
        raise InvalidPayload(msg)
    # Skip the attention-check trial: it has isCatch=True and a NaN price.
    real_trials = [t for t in trials if not t.get("isCatch", False)]
    if not real_trials:
        msg = f"{source}.task1.trials contained only catch trials"
        raise InvalidPayload(msg)
    P = np.array([float(_require(t, "price", f"{source}.task1.trials")) for t in real_trials])
    Q_obs = np.array(
        [float(_require(t, "quantity", f"{source}.task1.trials")) for t in real_trials]
    )

    task2 = _require(payload, "task2", source)
    per_fraction = _require(task2, "perFraction", f"{source}.task2")
    if not isinstance(per_fraction, list) or len(per_fraction) == 0:
        msg = f"{source}.task2.perFraction must be a non-empty list"
        raise InvalidPayload(msg)
    E_list: list[float] = []
    SV_list: list[float] = []
    for entry in per_fraction:
        fraction = float(_require(entry, "fraction", f"{source}.task2.perFraction"))
        titration = _require(entry, "titration", f"{source}.task2.perFraction")
        indifference = float(
            _require(titration, "indifference", f"{source}.task2.perFraction.titration")
        )
        E_list.append(fraction * p_max)
        SV_list.append(indifference)
    E = np.array(E_list, dtype=float)
    SV_obs = np.array(SV_list, dtype=float)

    return LoadedSession(
        session_id=session_id,
        arm=arm,
        p_max=p_max,
        P=P,
        Q_obs=Q_obs,
        E=E,
        SV_obs=SV_obs,
        raw=payload,
    )


def load_directory(path: Path | str, *, pattern: str = "*.json") -> LoadedCohort:
    """Load, parse, and stack every ``*.json`` in ``path`` (typically ``data/raw/``)."""
    p = Path(path)
    if not p.is_dir():
        msg = f"Not a directory: {p}"
        raise FileNotFoundError(msg)
    files = sorted(p.glob(pattern))
    if not files:
        msg = f"No files matching {pattern!r} in {p}"
        raise InvalidPayload(msg)
    payloads: list[LoadedSession] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            msg = f"{f}: invalid JSON: {exc}"
            raise InvalidPayload(msg) from exc
        payloads.append(parse_session(data, source=str(f)))
    return assemble_cohort(payloads)


def assemble_cohort(sessions: Iterable[LoadedSession]) -> LoadedCohort:
    """Stack a list of sessions into the matrix form. Validates shape consistency."""
    sessions_list = list(sessions)
    if not sessions_list:
        msg = "No sessions to assemble."
        raise InvalidPayload(msg)
    first = sessions_list[0]
    P = first.P
    n_prices = len(P)
    n_effort = len(first.E)
    for s in sessions_list[1:]:
        if not np.allclose(s.P, P):
            msg = (
                f"session {s.session_id}: price array differs from cohort baseline. "
                f"Pre-registration locks the price array; check experiment config."
            )
            raise InvalidPayload(msg)
        if len(s.E) != n_effort:
            msg = (
                f"session {s.session_id}: effort vector has length {len(s.E)}, expected {n_effort}"
            )
            raise InvalidPayload(msg)
    Q_obs = np.array([s.Q_obs for s in sessions_list], dtype=float)
    E = np.array([s.E for s in sessions_list], dtype=float)
    SV_obs = np.array([s.SV_obs for s in sessions_list], dtype=float)
    B_anchor = np.array([s.p_max for s in sessions_list], dtype=float)
    arm = np.array([s.arm for s in sessions_list], dtype=str)
    if Q_obs.shape != (len(sessions_list), n_prices):
        msg = (
            f"Q_obs shape {Q_obs.shape} disagrees with expected ({len(sessions_list)}, {n_prices})"
        )
        raise InvalidPayload(msg)
    return LoadedCohort(
        sessions=sessions_list,
        P=P,
        Q_obs=Q_obs,
        E=E,
        SV_obs=SV_obs,
        B_anchor=B_anchor,
        arm=arm,
    )


__all__ = [
    "InvalidPayload",
    "LoadedCohort",
    "LoadedSession",
    "assemble_cohort",
    "load_directory",
    "parse_session",
]
