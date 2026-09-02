"""Geography: city coordinate reference, great-circle distance, circuity.

The supplied coordinates are synthetic (Philadelphia is placed at -72.97 rather
than -75.16) but they are *internally consistent*: across all 60,000 supplied
rows every city carries exactly one (lat, lon) pair. That consistency is what
makes them usable — great-circle distance computed from them tracks the reported
road ``distance`` at r = 0.9995. They are treated as an embedding of the network,
not as real-world positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.7613

_PICKUP = ("pickup", "pickup_lat", "pickup_lon")
_DELIVERY = ("delivery", "delivery_lat", "delivery_lon")


def haversine_miles(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in miles between two coordinate arrays."""
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2))
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    inner = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def initial_bearing_degrees(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Initial compass bearing from origin to destination, in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2))
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


class InconsistentCityCoordinates(ValueError):
    """Raised when one city name maps to more than one coordinate pair."""


@dataclass(frozen=True)
class CityReference:
    """City name -> (lat, lon), built from every frame that carries coordinates.

    The December chart inputs ship city *names* only, so this table is how those
    rows acquire the geographic features the model was trained on.
    """

    table: pd.DataFrame

    @classmethod
    def from_frames(cls, *frames: pd.DataFrame) -> "CityReference":
        parts: list[pd.DataFrame] = []
        for frame in frames:
            for name_col, lat_col, lon_col in (_PICKUP, _DELIVERY):
                if not {name_col, lat_col, lon_col}.issubset(frame.columns):
                    continue
                part = frame[[name_col, lat_col, lon_col]].copy()
                part.columns = ["city", "lat", "lon"]
                parts.append(part)
        if not parts:
            raise ValueError("no frame supplied coordinate columns")

        stacked = pd.concat(parts, ignore_index=True).dropna(subset=["city", "lat", "lon"])
        stacked["lat"] = stacked["lat"].round(6)
        stacked["lon"] = stacked["lon"].round(6)

        spread = stacked.groupby("city")[["lat", "lon"]].nunique()
        offenders = spread[(spread["lat"] > 1) | (spread["lon"] > 1)]
        if not offenders.empty:
            raise InconsistentCityCoordinates(
                "cities with more than one coordinate pair: "
                f"{sorted(offenders.index.tolist())}"
            )

        table = (
            stacked.drop_duplicates(subset=["city"])
            .set_index("city")
            .sort_index()[["lat", "lon"]]
        )
        return cls(table=table)

    @property
    def cities(self) -> list[str]:
        return self.table.index.tolist()

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, city: str) -> bool:
        return city in self.table.index

    def lookup(self, cities: pd.Series) -> pd.DataFrame:
        """Return lat/lon for a Series of city names; unknown names give NaN."""
        joined = self.table.reindex(pd.Index(cities.astype(str)))
        joined.index = cities.index
        return joined

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill any missing coordinate columns from the reference table.

        Existing coordinates are preserved; only gaps are filled. This is what
        lets the December inputs and the training frame share one feature path.
        """
        result = frame.copy()
        for name_col, lat_col, lon_col in (_PICKUP, _DELIVERY):
            if name_col not in result.columns:
                continue
            looked_up = self.lookup(result[name_col])
            for target_col, source_col in ((lat_col, "lat"), (lon_col, "lon")):
                if target_col in result.columns:
                    result[target_col] = pd.to_numeric(
                        result[target_col], errors="coerce"
                    ).fillna(looked_up[source_col])
                else:
                    result[target_col] = looked_up[source_col]
        return result


def median_circuity(distance: pd.Series, great_circle: np.ndarray) -> float:
    """Typical road-to-great-circle ratio, robust to the corrupted rows.

    Uses the median so the 111 rows with an inflated ``distance`` cannot drag the
    repair constant they are about to be repaired with.
    """
    ratio = pd.Series(np.asarray(distance, dtype=float) / np.asarray(great_circle, dtype=float))
    return float(ratio.replace([np.inf, -np.inf], np.nan).dropna().median())
