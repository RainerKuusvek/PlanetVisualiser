import argparse
from dataclasses import dataclass
from math import cos, sin
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlanetConfig:
    radius_km: float
    orbital_period_days: float
    obliquity_deg: float
    precession_rate_deg_per_day: float


def _deg_to_rad(deg: float) -> float:
    return deg * np.pi / 180.0


def substellar_latitude_deg(
    t_days: np.ndarray, obliquity_deg: float, precession_rate_deg_per_day: float
) -> np.ndarray:
    """
    Simple model: the spin axis has fixed tilt (obliquity) that precesses around
    the orbital normal. This yields a sinusoidal substellar latitude.
    """
    omega = _deg_to_rad(precession_rate_deg_per_day)
    return obliquity_deg * np.cos(omega * t_days)


def solar_zenith_cosine(
    lat_deg: float, lon_deg: float, sub_lat_deg: np.ndarray, sub_lon_deg: float
) -> np.ndarray:
    lat = _deg_to_rad(lat_deg)
    lon = _deg_to_rad(lon_deg)
    sub_lat = _deg_to_rad(sub_lat_deg)
    sub_lon = _deg_to_rad(sub_lon_deg)
    return np.sin(lat) * np.sin(sub_lat) + np.cos(lat) * np.cos(sub_lat) * np.cos(
        lon - sub_lon
    )


def simulate_point_cycle(
    config: PlanetConfig,
    point_lat_deg: float,
    point_lon_deg: float,
    total_days: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_days = np.linspace(0.0, total_days, steps, dtype=float)

    # Tidally locked: substellar longitude is fixed in the body frame.
    sub_lon_deg = 0.0
    sub_lat_deg = substellar_latitude_deg(
        t_days, config.obliquity_deg, config.precession_rate_deg_per_day
    )
    cos_zenith = solar_zenith_cosine(point_lat_deg, point_lon_deg, sub_lat_deg, sub_lon_deg)
    return t_days, sub_lat_deg, cos_zenith


def plot_cycle(
    t_days: np.ndarray,
    sub_lat_deg: np.ndarray,
    cos_zenith: np.ndarray,
    point_lat_deg: float,
    point_lon_deg: float,
) -> None:
    is_day = cos_zenith > 0.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t_days, sub_lat_deg, color="#1f77b4")
    axes[0].set_ylabel("Substellar latitude (deg)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_days, cos_zenith, color="#ff7f0e", label="cos(zenith)")
    axes[1].fill_between(
        t_days, 0.0, cos_zenith, where=is_day, color="#ffd27f", alpha=0.4, label="day"
    )
    axes[1].axhline(0.0, color="#444", linewidth=1.0)
    axes[1].set_ylabel("cos(zenith)")
    axes[1].set_xlabel("Time (days)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        f"Day/Night Cycle at lat {point_lat_deg:.1f}°, lon {point_lon_deg:.1f}°"
    )
    fig.tight_layout()
    plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize day/night cycle for a tidally-locked planet with precession."
    )
    parser.add_argument("--radius-km", type=float, default=6371.0)
    parser.add_argument("--orbital-period-days", type=float, default=10.0)
    parser.add_argument("--obliquity-deg", type=float, default=5.0)
    parser.add_argument("--precession-rate-deg-per-day", type=float, default=None)
    parser.add_argument("--precession-period-days", type=float, default=200.0)
    parser.add_argument("--point-lat-deg", type=float, default=0.0)
    parser.add_argument("--point-lon-deg", type=float, default=0.0)
    parser.add_argument("--total-days", type=float, default=400.0)
    parser.add_argument("--steps", type=int, default=1000)
    return parser


def _resolve_precession_rate(
    rate_deg_per_day: float | None, period_days: float | None
) -> float:
    if rate_deg_per_day is not None and period_days is not None:
        raise ValueError(
            "Provide either --precession-rate-deg-per-day or --precession-period-days, not both."
        )
    if rate_deg_per_day is None and period_days is None:
        raise ValueError("Provide --precession-rate-deg-per-day or --precession-period-days.")
    if rate_deg_per_day is not None:
        return rate_deg_per_day
    if period_days <= 0.0:
        raise ValueError("--precession-period-days must be positive.")
    return 360.0 / period_days


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    precession_rate = _resolve_precession_rate(
        args.precession_rate_deg_per_day, args.precession_period_days
    )

    config = PlanetConfig(
        radius_km=args.radius_km,
        orbital_period_days=args.orbital_period_days,
        obliquity_deg=args.obliquity_deg,
        precession_rate_deg_per_day=precession_rate,
    )

    t_days, sub_lat_deg, cos_zenith = simulate_point_cycle(
        config=config,
        point_lat_deg=args.point_lat_deg,
        point_lon_deg=args.point_lon_deg,
        total_days=args.total_days,
        steps=args.steps,
    )
    plot_cycle(t_days, sub_lat_deg, cos_zenith, args.point_lat_deg, args.point_lon_deg)


if __name__ == "__main__":
    main()
