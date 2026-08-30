"""Time handling for the LLM4POI preprocessed files.

`UTCTimeOffsetEpoch` is unusable: it is `UTCTimeOffset` passed through a naive
`datetime.timestamp()` on a UTC+10/+11 machine, so the same offset appears for
New York, Tokyo and California alike, and hour-of-day read from it places Tokyo's
quietest hour at 17:00.

`UTCTimeOffset` is used instead. Its meaning differs per dataset:
  nyc, tky : local wall clock, offset applied (Foursquare).
  ca       : raw UTC, offset not applied (Gowalla).

`assert_human_rhythm` checks the result: under the correct reading each city
shows a 3-5am trough and a daytime peak.

Two quantities come out of this module and are not interchangeable:
  ts_utc       absolute instant, for elapsed time between check-ins.
  hour / dow   local calendar position, for the periodic features.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# tz            IANA zone of the city.
# column_means  what the `UTCTimeOffset` column actually holds.
TIME_SPEC: Dict[str, Dict[str, str]] = {
    "nyc": {"tz": "America/New_York", "column_means": "local"},
    "tky": {"tz": "Asia/Tokyo", "column_means": "local"},
    "ca": {"tz": "America/Los_Angeles", "column_means": "utc"},
}


class TimeParseError(RuntimeError):
    pass


def parse_times(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add ts_utc (int seconds), local_hour (float 0-24), local_dow (0=Mon).

    `UTCTimeOffsetEpoch` is ignored entirely. It is never read by this package.
    """
    ds = dataset.lower()
    if ds not in TIME_SPEC:
        raise TimeParseError(f"no time spec for dataset {ds!r}")
    spec = TIME_SPEC[ds]

    if "UTCTimeOffset" not in df.columns:
        raise TimeParseError(
            f"{ds}: column 'UTCTimeOffset' is missing. The corrupt "
            f"'UTCTimeOffsetEpoch' column is not an acceptable substitute.")

    naive = pd.to_datetime(df["UTCTimeOffset"])
    if naive.isna().any():
        raise TimeParseError(f"{ds}: {naive.isna().sum()} unparseable timestamps")

    if spec["column_means"] == "local":
        # Wall clock in the city. Localize to recover the instant. DST fall-back
        # hours are ambiguous; resolve to standard time and shift nonexistent
        # spring-forward times rather than dropping check-ins.
        aware_local = naive.dt.tz_localize(spec["tz"], ambiguous=False, nonexistent="shift_forward")
        utc = aware_local.dt.tz_convert("UTC")
        local = aware_local
    else:
        utc = naive.dt.tz_localize("UTC")
        local = utc.dt.tz_convert(spec["tz"])

    out = df.copy()
    out["ts_utc"] = (utc.astype("int64") // 10 ** 9).astype("int64")
    out["local_hour"] = (local.dt.hour + local.dt.minute / 60.0 + local.dt.second / 3600.0).astype("float32")
    out["local_dow"] = local.dt.dayofweek.astype("int8")  # 0=Monday
    return out


def rhythm_stats(local_hour: np.ndarray) -> Dict[str, float]:
    """Shape of the daily check-in rhythm."""
    h = np.asarray(local_hour)
    dist = np.bincount(h.astype(int).clip(0, 23), minlength=24) / max(len(h), 1)
    night = float(dist[3:6].sum())
    day = float(dist[11:22].sum())
    return {
        "night_3_6_frac": night,
        "day_11_22_frac": day,
        "day_night_ratio": day / max(night, 1e-9),
        "argmin_hour": int(dist.argmin()),
        "argmax_hour": int(dist.argmax()),
        "hist": dist.tolist(),
    }


def assert_human_rhythm(local_hour: np.ndarray, dataset: str, min_ratio: float = 3.0) -> Dict[str, float]:
    """Raise if the parsed local time is not a plausible human rhythm.

    A guard, not a formality. Every wrong reading of these columns that we found
    puts the daily minimum somewhere between 11:00 and 17:00. Real check-ins
    trough between 03:00 and 06:00.
    """
    st = rhythm_stats(local_hour)
    bad_min = not (1 <= st["argmin_hour"] <= 7)
    bad_ratio = st["day_night_ratio"] < min_ratio
    if bad_min or bad_ratio:
        raise TimeParseError(
            f"{dataset}: parsed local time does not look like human behaviour "
            f"(quietest hour = {st['argmin_hour']:02d}:00, busiest = {st['argmax_hour']:02d}:00, "
            f"day/night ratio = {st['day_night_ratio']:.1f}). Expected the trough between "
            f"03:00 and 06:00. The time column is being read wrong; check TIME_SPEC[{dataset!r}].")
    return st
