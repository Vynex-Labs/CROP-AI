"""Load weather/trap/observation JSONL and build SYNTHETIC smoke panels."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from cropai.dataset.schema import ObservationRecord, TrapRecord, WeatherRecord, write_jsonl
from cropai.risk.schema import RiskRequest
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import data_dir


def _records(path: Path, cls):
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            known = {f.name for f in cls.__dataclass_fields__.values()} if hasattr(cls, "__dataclass_fields__") else payload.keys()
            out.append(cls(**{k: v for k, v in payload.items() if k in known}))
    return out


def load_weather(path: Path) -> list[WeatherRecord]:
    return _records(path, WeatherRecord)


def load_traps(path: Path) -> list[TrapRecord]:
    return _records(path, TrapRecord)


def load_observations(path: Path) -> list[ObservationRecord]:
    return _records(path, ObservationRecord)


def request_from_files(
    *,
    crop_id: str,
    weather_path: Path | None = None,
    trap_path: Path | None = None,
    obs_path: Path | None = None,
    timestamp: str = "",
    lat: float | None = None,
    lon: float | None = None,
    growth_stage: str = "unknown",
    planting_date: str = "",
    is_synthetic: bool = False,
) -> RiskRequest:
    return RiskRequest(
        crop_id=crop_id,
        timestamp=timestamp or utc_now_iso(),
        lat=lat,
        lon=lon,
        growth_stage=growth_stage,
        planting_date=planting_date,
        weather=load_weather(weather_path) if weather_path else [],
        traps=load_traps(trap_path) if trap_path else [],
        observations=load_observations(obs_path) if obs_path else [],
        is_synthetic=is_synthetic
        or bool(weather_path and "synthetic" in str(weather_path)),
    )


def generate_synthetic_weather_series(
    out_dir: Path | None = None,
    *,
    days: int = 14,
    seed: int = 42,
    as_of: date | None = None,
) -> dict[str, Path]:
    """Labeled SYNTHETIC daily series for pipeline tests. Not real weather."""
    import random

    rng = random.Random(seed)
    as_of = as_of or date(2026, 8, 31)
    out_dir = out_dir or (data_dir() / "synthetic" / "risk")
    out_dir.mkdir(parents=True, exist_ok=True)
    weather: list[WeatherRecord] = []
    traps: list[TrapRecord] = []
    obs: list[ObservationRecord] = []
    for i in range(days):
        d = as_of - timedelta(days=days - 1 - i)
        ts = f"{d.isoformat()}T12:00:00+00:00"
        wet = i % 4 == 0
        weather.append(
            WeatherRecord(
                station_id="syn_wx_risk",
                timestamp=ts,
                lat=19.0,
                lon=74.0,
                temperature_c=round(22 + rng.random() * 10, 2),
                humidity_pct=round((85 if wet else 55) + rng.random() * 8, 2),
                rainfall_mm=round((18 if wet else 0) + rng.random(), 2) if wet else 0.0,
                wind_ms=round(1 + rng.random() * 3, 2),
                soil_moisture=None,
                is_synthetic=True,
                source="synthetic_risk_smoke",
                missing_fields="soil_moisture",
            )
        )
        traps.append(
            TrapRecord(
                trap_id="syn_trap_risk",
                timestamp=ts,
                trap_type="pheromone",
                pest_id="brown_planthopper",
                count=int(rng.randint(0, 4) + (6 if wet else 0)),
                crop_id="rice",
                lat=19.0,
                lon=74.0,
                is_synthetic=True,
                source="synthetic_risk_smoke",
            )
        )
        if wet and i > 8:
            obs.append(
                ObservationRecord(
                    observation_id=f"syn_obs_risk_{i:02d}",
                    timestamp=ts,
                    crop_id="rice",
                    farm_id="syn_farm_risk",
                    field_id="syn_field_risk",
                    lat=19.0,
                    lon=74.0,
                    disease_id="rice_blast",
                    severity_bin="early",
                    source="synthetic",
                    is_synthetic=True,
                    notes="synthetic_risk_smoke",
                )
            )
    w_path = out_dir / "weather.jsonl"
    t_path = out_dir / "traps.jsonl"
    o_path = out_dir / "observations.jsonl"
    write_jsonl(w_path, weather)
    write_jsonl(t_path, traps)
    write_jsonl(o_path, obs)
    (out_dir / "SYNTHETIC.txt").write_text(
        "SYNTHETIC weather/trap/observation series for risk pipeline tests. "
        "Not real meteorological or outbreak data.\n",
        encoding="utf-8",
    )
    return {"weather": w_path, "traps": t_path, "observations": o_path, "dir": out_dir}


def count_real_outbreak_labels(paths: Iterable[Path]) -> int:
    """Count observation rows that are not synthetic. Honest zero if none."""
    n = 0
    for path in paths:
        if not path.exists():
            continue
        for row in load_observations(path):
            if not row.is_synthetic and row.disease_id and not str(row.disease_id).endswith("_healthy"):
                n += 1
    return n
