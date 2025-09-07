from __future__ import annotations

from typing import Dict, List
import pandas as pd

DATE_CANDIDATES = [
    "tarih", "date", "gun", "day", "zaman", "time",
]

TARGET_CANDIDATES = [
    "satis", "sales", "miktar", "quantity", "adet", "amount", "value", "satis_miktari",
]


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Tarih ve hedef kolonlarını akıllı tespit eder."""
    lower_map = {c.lower(): c for c in df.columns}

    def find(candidates: List[str], is_date: bool = False) -> str:
        # 1) alias eşleşmesi
        for alias in candidates:
            if alias in lower_map:
                return lower_map[alias]
        # 2) tarih için fallback: datetime parse edilebilen ilk kolon
        if is_date:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c])
                    return c
                except Exception:
                    continue
        raise ValueError("Zorunlu kolon bulunamadı")

    date_col = find(DATE_CANDIDATES, is_date=True)
    target_col = find(TARGET_CANDIDATES, is_date=False)
    return {"date": date_col, "target": target_col}


def normalize_schema(df: pd.DataFrame, mapping: Dict[str, str], feature_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={mapping["date"]: "ds", mapping["target"]: "y"})
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds", "y"])
    kept = [c for c in (feature_columns or []) if c in out.columns]
    return out[["ds", "y", *kept]].sort_values("ds").reset_index(drop=True)


def aggregate_time_series(
    df: pd.DataFrame,
    level: str,
    weekly_rule: str = "W-MON",
    monthly_rule: str = "MS",
) -> pd.DataFrame:
    """
    Normalize schema'dan gelen df'yi tek ankrajla toplar.
    - weekly: Pazartesi başlangıç (W-MON)
    - monthly: Ay başı (MS)
    """
    if level not in {"weekly", "monthly"}:
        raise ValueError("Geçersiz aggregation_level")
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    rule = weekly_rule if level == "weekly" else monthly_rule
    grouped = out.set_index("ds")["y"].resample(rule).sum().reset_index()
    grouped.columns = ["ds", "y"]
    return grouped.sort_values("ds").reset_index(drop=True)


def sanitize_outliers_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Eksik zaman noktalarını frekansına göre doldurur.
    - IQR clipping ile aykırı değerleri sınırlar.
    """
    out = df.copy()
    if len(out) < 2:
        return out

    out["ds"] = pd.to_datetime(out["ds"])
    inferred = pd.infer_freq(out["ds"])
    if not inferred:
        inferred = _infer_by_gap(out)  # haftalıkta W-MON'a yönelir

    full_index = pd.date_range(start=out["ds"].min(), end=out["ds"].max(), freq=inferred)
    out = out.set_index("ds").reindex(full_index)
    out.index.name = "ds"

    # basit imputasyon
    out["y"] = out["y"].ffill().bfill()

    # IQR clipping
    q1, q3 = out["y"].quantile(0.25), out["y"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out["y"] = out["y"].clip(lower=lower, upper=upper)

    return out.reset_index().rename(columns={"index": "ds"})


def _infer_by_gap(df: pd.DataFrame) -> str:
    """Aralıktan frekans çıkarımı: haftalıkta W-MON varsayılanına yönel."""
    deltas = df["ds"].diff().dropna()
    if len(deltas) == 0:
        return "W-MON"
    median_days = deltas.median().days
    if median_days >= 25:
        return "MS"
    if median_days >= 6:
        return "W-MON"
    # bu API’de günlük mod yok; haftalığa sabitle
    return "W-MON"
