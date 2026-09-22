# place_database.py
"""
Place database for the no-SP pipeline.

Stores one entry per visited place. Each entry contains:
    - The SDR produced by the encoder (for place matching).
    - A list of angular templates (yaw clusters) seen at this place.
    - Visit counts and timestamps.

Place matching is done by SDR overlap: a new cloud is assigned to the
existing place with the highest overlap, if that overlap exceeds a
threshold. Otherwise a new place is created.

Angular templates are stored as clusters. When a new cloud is matched to
an existing place, its estimated yaw is compared against existing
templates: if it is close to one (within `yaw_tolerance_deg`), that
template's count is incremented; otherwise a new template is added.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# Data classes
# ============================================================
@dataclass
class AngularTemplate:
    """A yaw cluster seen at a given place."""
    mean_yaw_rad: float        # representative yaw (circular mean)
    visit_count: int = 1
    last_seen_ts: float = field(default_factory=time.time)
    # Optional: store the angular histogram for later refinement
    angular_histogram: Optional[np.ndarray] = None

    def to_json(self) -> dict:
        d = {
            "mean_yaw_rad":     self.mean_yaw_rad,
            "visit_count":      self.visit_count,
            "last_seen_ts":     self.last_seen_ts,
        }
        if self.angular_histogram is not None:
            d["angular_histogram"] = self.angular_histogram.tolist()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "AngularTemplate":
        hist = d.get("angular_histogram")
        return cls(
            mean_yaw_rad=d["mean_yaw_rad"],
            visit_count=d["visit_count"],
            last_seen_ts=d["last_seen_ts"],
            angular_histogram=np.array(hist) if hist is not None else None,
        )


@dataclass
class PlaceEntry:
    """A single place in the database."""
    place_id: int
    sdr: np.ndarray                          # (total_size,) 0/1
    angular_templates: List[AngularTemplate] = field(default_factory=list)
    visit_count: int = 1
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    label: str = ""                          # optional human-readable label

    @property
    def n_templates(self) -> int:
        return len(self.angular_templates)

    def to_json(self) -> dict:
        return {
            "place_id":          self.place_id,
            "visit_count":       self.visit_count,
            "first_seen_ts":     self.first_seen_ts,
            "last_seen_ts":      self.last_seen_ts,
            "label":             self.label,
            "angular_templates": [t.to_json() for t in self.angular_templates],
            # sdr is stored separately in a .npy file
        }


# ============================================================
# Helper: circular angle utilities
# ============================================================
def circular_mean(angles_rad: List[float]) -> float:
    """Circular mean of a list of angles (radians), robust to wraparound."""
    s = np.mean(np.sin(angles_rad))
    c = np.mean(np.cos(angles_rad))
    return float(np.arctan2(s, c))


def angular_distance(a_rad: float, b_rad: float) -> float:
    """Shortest angular distance between two angles (radians)."""
    d = abs(a_rad - b_rad) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))


# ============================================================
# PlaceDatabase
# ============================================================
class PlaceDatabase:
    """
    Manages a set of visited places.

    Public API
    ----------
    match_or_create(sdr, yaw_rad, angular_hist) -> (place_id, matched, template_idx)
        Match a new observation to an existing place (or create a new one).
        If matched to a place, updates its angular templates.
        Returns:
            place_id     : int
            matched      : bool  (True if matched, False if new place)
            template_idx : int   (-1 if new place; otherwise the template matched)

    get_sdr(place_id) -> np.ndarray
    get_templates(place_id) -> List[AngularTemplate]
    save(path_prefix) / load(path_prefix)
    """

    def __init__(
        self,
        sdr_size: int,
        match_threshold: float = 0.5,
        yaw_tolerance_deg: float = 15.0,
    ):
        """
        Parameters
        ----------
        sdr_size : int
            Length of the encoder SDR (e.g., 56000).
        match_threshold : float
            Minimum overlap ratio (overlap / active_bits) to consider two
            SDRs as the same place. Typical: 0.5 (50%).
        yaw_tolerance_deg : float
            Maximum angular distance (degrees) to consider two yaws as the
            same template. Typical: 15 degrees.
        """
        self.sdr_size = int(sdr_size)
        self.match_threshold = float(match_threshold)
        self.yaw_tolerance_rad = float(np.deg2rad(yaw_tolerance_deg))

        self.places: List[PlaceEntry] = []
        self._next_place_id: int = 0

        # Stats (useful for diagnostics)
        self.n_queries: int = 0
        self.n_matches: int = 0
        self.n_creations: int = 0

    # ---------- Core API ----------
    def match_or_create(
        self,
        sdr: np.ndarray,
        yaw_rad: float,
        angular_hist: Optional[np.ndarray] = None,
        label: str = "",
    ) -> Tuple[int, bool, int]:
        """
        Match an observation against the database.

        Returns
        -------
        place_id : int
        matched : bool
        template_idx : int  (-1 if new place, else index of the matched template)
        """
        self.n_queries += 1

        # ---- 1. Find best-matching place by SDR overlap ----
        best_place_idx = -1
        best_overlap_ratio = 0.0
        sdr_active = int(sdr.sum())

        if sdr_active == 0:
            raise ValueError("Empty SDR (no active bits)")

        for idx, place in enumerate(self.places):
            overlap = int(np.logical_and(sdr, place.sdr).sum())
            ratio = overlap / sdr_active
            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_place_idx = idx

        # ---- 2. Decide match vs. create ----
        matched = best_overlap_ratio >= self.match_threshold

        print(f"Query: overlap={best_overlap_ratio:.3f}, idx={best_place_idx}, "  
              f"matched={'yes' if matched else 'no'}")

        if not matched:
            # Create new place
            place_id = self._next_place_id
            self._next_place_id += 1

            new_place = PlaceEntry(
                place_id=place_id,
                sdr=sdr.copy(),
                angular_templates=[
                    AngularTemplate(
                        mean_yaw_rad=yaw_rad,
                        visit_count=1,
                        angular_histogram=angular_hist,
                    )
                ],
                visit_count=1,
                label=label,
            )
            self.places.append(new_place)
            self.n_creations += 1
            return place_id, False, -1

        # ---- 3. Matched: update place and angular templates ----
        place = self.places[best_place_idx]
        place.visit_count += 1
        place.last_seen_ts = time.time()

        # Find nearest angular template
        best_t_idx = -1
        best_t_dist = np.inf
        for t_idx, t in enumerate(place.angular_templates):
            dist = angular_distance(yaw_rad, t.mean_yaw_rad)
            if dist < best_t_dist:
                best_t_dist = dist
                best_t_idx = t_idx

        if best_t_dist <= self.yaw_tolerance_rad:
            # Update existing template
            t = place.angular_templates[best_t_idx]
            # Circular mean update: recompute from old mean weighted by count
            # (approximation; more accurate would be to store all angles)
            w_old = t.visit_count
            w_new = 1
            s = (w_old * np.sin(t.mean_yaw_rad) + w_new * np.sin(yaw_rad)) / (w_old + w_new)
            c = (w_old * np.cos(t.mean_yaw_rad) + w_new * np.cos(yaw_rad)) / (w_old + w_new)
            t.mean_yaw_rad = float(np.arctan2(s, c))
            t.visit_count += 1
            t.last_seen_ts = time.time()
            if angular_hist is not None and t.angular_histogram is not None:
                # Running average of histograms
                t.angular_histogram = (
                    (w_old * t.angular_histogram + w_new * angular_hist)
                    / (w_old + w_new)
                )
            template_idx = best_t_idx
        else:
            # Add a new template to this place
            place.angular_templates.append(
                AngularTemplate(
                    mean_yaw_rad=yaw_rad,
                    visit_count=1,
                    angular_histogram=angular_hist,
                )
            )
            template_idx = len(place.angular_templates) - 1

        self.n_matches += 1
        return place.place_id, True, template_idx

    # ---------- Accessors ----------
    def get_place(self, place_id: int) -> Optional[PlaceEntry]:
        for p in self.places:
            if p.place_id == place_id:
                return p
        return None

    def get_sdr(self, place_id: int) -> Optional[np.ndarray]:
        p = self.get_place(place_id)
        return p.sdr if p is not None else None

    def get_templates(self, place_id: int) -> List[AngularTemplate]:
        p = self.get_place(place_id)
        return p.angular_templates if p is not None else []

    def summary(self) -> str:
        lines = [
            f"PlaceDatabase: {len(self.places)} places, "
            f"{self.n_queries} queries, "
            f"{self.n_matches} matches, "
            f"{self.n_creations} creations",
        ]
        for p in self.places:
            yaws_deg = [f"{np.rad2deg(t.mean_yaw_rad):+.0f}°"
                        for t in p.angular_templates]
            lines.append(
                f"  place {p.place_id:>3}: "
                f"visits={p.visit_count:>3}, "
                f"templates={p.n_templates:>2} "
                f"[{', '.join(yaws_deg)}]"
                + (f"  # {p.label}" if p.label else "")
            )
        return "\n".join(lines)

    # ---------- Persistence ----------
    def save(self, prefix: str | Path) -> None:
        """
        Save the database to two files:
            <prefix>_meta.json    - all metadata + angular templates
            <prefix>_sdrs.npy     - (N_places, sdr_size) array of SDRs
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "sdr_size":         self.sdr_size,
            "match_threshold":  self.match_threshold,
            "yaw_tolerance_rad": self.yaw_tolerance_rad,
            "next_place_id":    self._next_place_id,
            "n_queries":        self.n_queries,
            "n_matches":        self.n_matches,
            "n_creations":      self.n_creations,
            "places":           [p.to_json() for p in self.places],
        }
        prefix.with_name(prefix.name + "_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

        # Stack SDRs into one array
        if self.places:
            sdrs = np.stack([p.sdr for p in self.places], axis=0)
        else:
            sdrs = np.zeros((0, self.sdr_size), dtype=np.uint8)
        np.save(prefix.with_name(prefix.name + "_sdrs.npy"), sdrs)

    @classmethod
    def load(cls, prefix: str | Path) -> "PlaceDatabase":
        prefix = Path(prefix)
        meta = json.loads(
            prefix.with_name(prefix.name + "_meta.json").read_text()
        )
        sdrs = np.load(prefix.with_name(prefix.name + "_sdrs.npy"))

        db = cls(
            sdr_size=meta["sdr_size"],
            match_threshold=meta["match_threshold"],
            yaw_tolerance_deg=float(np.rad2deg(meta["yaw_tolerance_rad"])),
        )
        db._next_place_id = meta["next_place_id"]
        db.n_queries      = meta["n_queries"]
        db.n_matches      = meta["n_matches"]
        db.n_creations    = meta["n_creations"]

        for i, pj in enumerate(meta["places"]):
            place = PlaceEntry(
                place_id=pj["place_id"],
                sdr=sdrs[i],
                angular_templates=[
                    AngularTemplate.from_json(tj) for tj in pj["angular_templates"]
                ],
                visit_count=pj["visit_count"],
                first_seen_ts=pj["first_seen_ts"],
                last_seen_ts=pj["last_seen_ts"],
                label=pj["label"],
            )
            db.places.append(place)
        return db