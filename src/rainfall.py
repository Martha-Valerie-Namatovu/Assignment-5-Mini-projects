"""Mini-Project 4: rainfall pattern & crop suitability analyser.

Rainfall is monthly totals in mm, January..December.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class Region:
    """A region and its 12-month rainfall climatology."""

    def __init__(self, name: str, rainfall_mm: ArrayLike) -> None:
        arr = np.asarray(rainfall_mm, dtype=float)
        if arr.shape != (12,):
            raise ValueError(f"rainfall must have exactly 12 monthly values, got shape {arr.shape}")
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("rainfall must be finite and non-negative")
        self.name = name
        self.rainfall = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm)"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Sample SD / mean: how unevenly rain is spread through the year."""
        m = self.mean()
        if m == 0:
            raise ValueError("CV undefined for a region with zero rainfall")
        return float(self.rainfall.std(ddof=1) / m)

    def rainfall_peaks(self, prominence: float = 10.0) -> np.ndarray:
        """Indices (0=Jan) of local rainfall maxima from ``scipy.signal.find_peaks``.

        The year is circular, so the series is tiled three times and only peaks in
        the middle copy are kept; otherwise a December/January peak is lost at the
        array edge. `prominence` (mm) filters month-to-month noise.
        """
        tiled = np.tile(self.rainfall, 3)
        peaks, _ = find_peaks(tiled, prominence=prominence)
        return np.sort(peaks[(peaks >= 12) & (peaks < 24)] - 12)

    def rainy_seasons(self, prominence: float = 10.0, dry_threshold: float | None = None) -> np.ndarray:
        """Peaks that belong to genuinely separate rainy seasons.

        Two neighbouring peaks (going round the calendar) are separate seasons only
        if the driest month between them is a *dry spell*: below ``dry_threshold``
        mm, which defaults to the region's own mean monthly rainfall (so "dry" is
        relative to local climate). A shallow mid-season dip that stays above the
        mean (e.g. Gulu in June) does not split a season. Merged peaks keep the higher one.
        """
        threshold = self.mean() if dry_threshold is None else float(dry_threshold)
        peaks = list(self.rainfall_peaks(prominence))
        r = self.rainfall
        merged = True
        while merged and len(peaks) > 1:
            merged = False
            for i in range(len(peaks)):
                a, b = peaks[i], peaks[(i + 1) % len(peaks)]
                gap = [(a + k) % 12 for k in range(1, (b - a) % 12 or 12)]
                if not gap or min(r[m] for m in gap) >= threshold:
                    keep = a if r[a] >= r[b] else b
                    peaks = sorted({p for p in peaks if p not in (a, b)} | {keep})
                    merged = True
                    break
        return np.array(sorted(peaks), dtype=int)

    def modality(self, prominence: float = 10.0, dry_threshold: float | None = None) -> str:
        n = len(self.rainy_seasons(prominence, dry_threshold))
        return {0: "aseasonal", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


@dataclass
class CropRule:
    """Monthly rainfall window (mm) in which a crop's water needs are met.

    Below `min_mm` → drought stress; above `max_mm` → waterlogging / disease risk.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""
    notes: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.min_mm < 0 or self.max_mm <= self.min_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    def classify(self, rainfall_mm: float) -> str:
        if rainfall_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rainfall_mm < self.min_mm:
            return "Drought risk"
        if rainfall_mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def score(self, rainfall_mm: float) -> int:
        """-1 drought, 0 suitable, +1 waterlogging (convenient for heatmaps)."""
        label = self.classify(rainfall_mm)
        return -1 if label == "Drought risk" else (1 if label == "Waterlogging risk" else 0)


def classify_region(region: Region, rules: list[CropRule]) -> dict[str, list[str]]:
    """Label every month of a region for every crop."""
    return {rule.crop: [rule.classify(v) for v in region.rainfall] for rule in rules}


# ---- similarity / distance -------------------------------------------------
def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """dot(a, b) / (‖a‖·‖b‖). Measures the angle between vectors, i.e. *shape*, not size."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / (na * nb))


def pearson(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation = cosine similarity of the mean-centred vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return cosine_similarity(a - a.mean(), b - b.mean())


def euclidean(a: ArrayLike, b: ArrayLike) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same length")
    return float(np.linalg.norm(a - b))


def pairwise_matrix(regions: list[Region], metric) -> np.ndarray:
    """Symmetric matrix of `metric(region_i.rainfall, region_j.rainfall)`."""
    n = len(regions)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = metric(regions[i].rainfall, regions[j].rainfall)
    return out


# ---- real data (extension): NASA POWER monthly JSON --------------------------
def parse_power_monthly(payload: dict, parameter: str = "PRECTOTCORR") -> dict[int, np.ndarray]:
    """Turn a NASA POWER *monthly point* JSON response into {year: 12 monthly totals in mm}.

    The API stores values under ``properties.parameter.<PARAM>`` with keys ``YYYYMM``;
    month ``13`` is the annual value and is skipped. Missing values use the fill
    value -999 and become NaN. If the reported units are a daily rate (mm/day),
    values are multiplied by the number of days in each month to give monthly totals.
    """
    import calendar

    try:
        series = payload["properties"]["parameter"][parameter]
    except KeyError as err:
        raise ValueError(f"payload has no properties.parameter.{parameter}") from err
    units = str(payload.get("parameters", {}).get(parameter, {}).get("units", "mm/day")).lower()
    per_day = "day" in units
    out: dict[int, np.ndarray] = {}
    for key, value in series.items():
        year, month = int(str(key)[:4]), int(str(key)[4:6])
        if not 1 <= month <= 12:
            continue
        v = np.nan if value is None or float(value) <= -999 else float(value)
        if per_day and np.isfinite(v):
            v *= calendar.monthrange(year, month)[1]
        out.setdefault(year, np.full(12, np.nan))[month - 1] = v
    if not out:
        raise ValueError("no monthly values found")
    return dict(sorted(out.items()))
