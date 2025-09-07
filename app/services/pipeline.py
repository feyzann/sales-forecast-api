from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .preprocess import (
    detect_columns,
    normalize_schema,
    aggregate_time_series,
    sanitize_outliers_and_missing,
)
from .predictor import EnsembleForecaster


def _to_iso_date(value: Any) -> str:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, )):
        return value.date().isoformat()
    try:
        ts = pd.to_datetime(value)
        return ts.date().isoformat()
    except Exception:
        s = str(value).strip()
        # kaba bir güvenli fallback
        return s[:10]


@dataclass
class PredictionPipeline:
    prediction_frequency: str          # "weekly" | "monthly"
    aggregation_level: str             # "weekly" | "monthly"
    prediction_period: int
    feature_columns: List[str]
    return_confidence: bool
    min_data_points: int               # post-aggregation min points
    weekly_rule: str = "W-MON"         # tek ankraj: Pazartesi
    monthly_rule: str = "MS"           # tek ankraj: Ay başı
    non_negative: bool = True          # negatif tahminleri kırp

    def run(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame.from_records(records)
        if df.empty:
            raise ValueError("Boş veri kümesi")

        # 1) Kolon tespiti + normalize
        detected = detect_columns(df)
        df_norm = normalize_schema(df, detected, self.feature_columns)

        # 2) Toplama (tek ankraj ile)
        df_agg = aggregate_time_series(
            df_norm,
            level=self.aggregation_level,
            weekly_rule=self.weekly_rule,
            monthly_rule=self.monthly_rule,
        )

        # 3) Temizlik (eksikler + aykırı değer clipping)
        df_clean = sanitize_outliers_and_missing(df_agg)

        # 4) Min nokta kontrolü (toplamadan sonra)
        if len(df_clean) < self.min_data_points:
            raise ValueError("Güvenilir tahmin için en az 30 veri noktası gereklidir.")

        # 5) Modeli çalıştır
        forecaster = EnsembleForecaster(
            frequency=self.prediction_frequency,
            horizon=self.prediction_period,
            return_confidence=self.return_confidence,
            weekly_rule=self.weekly_rule,
            monthly_rule=self.monthly_rule,
        )
        forecast_df, model_info, data_summary = forecaster.fit_predict(df_clean)

        # 6) data_summary fallback/normalize
        # df_clean beklenen şema: columns=["ds","y", ...]
        try:
            ds_min = pd.to_datetime(df_clean["ds"].min()).date().isoformat()
            ds_max = pd.to_datetime(df_clean["ds"].max()).date().isoformat()
            date_range = f"{ds_min} to {ds_max}"
        except Exception:
            date_range = None

        features_used = data_summary.get("features_used") if isinstance(data_summary, dict) else None
        if not features_used:
            # Prophet default + yıllık sezonsellik
            features_used = ["trend", "weekly_seasonality", "yearly_seasonality"]

        data_summary_out = {
            "input_rows": int(data_summary.get("input_rows")) if isinstance(data_summary, dict) and "input_rows" in data_summary else int(len(df_clean)),
            "date_range": data_summary.get("date_range", date_range) if isinstance(data_summary, dict) else date_range,
            "features_used": features_used,
        }

        # 7) Tahminleri normalize et (ISO tarih, negatif klip)
        preds: List[Dict[str, Any]] = []
        if not forecast_df.empty:
            # sıralı dön
            forecast_df = forecast_df.sort_values("ds")
            for _, row in forecast_df.iterrows():
                yhat = float(row.get("yhat", 0.0))
                lo = row.get("yhat_lower", None)
                hi = row.get("yhat_upper", None)

                if self.non_negative:
                    yhat = max(0.0, yhat)
                    if lo is not None:
                        lo = max(0.0, float(lo))
                    if hi is not None:
                        hi = max(0.0, float(hi))

                item = {
                    "date": _to_iso_date(row.get("ds")),
                    "predicted_value": yhat,
                }
                if self.return_confidence and {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
                    item["confidence_lower"] = lo if lo is not None else None
                    item["confidence_upper"] = hi if hi is not None else None

                preds.append(item)

        # 8) model_info normalize (etiket)
        mi = model_info.copy() if isinstance(model_info, dict) else {}
        algo = mi.get("algorithm")
        if not algo:
            algo = "Prophet"
        # Eğer EnsembleForecaster sadece Prophet içeriyorsa etiketi sadeleştir
        if str(algo).strip().lower().startswith("ensemble") and "xgboost" not in str(algo).lower():
            algo = "Prophet"
        mi["algorithm"] = algo

        result: Dict[str, Any] = {
            "success": True,
            "data_summary": data_summary_out,
            "predictions": preds,
            "model_info": mi,
        }
        return result


def build_pipeline(
    prediction_frequency: str,
    aggregation_level: str,
    prediction_period: int,
    feature_columns: Optional[List[str]],
    return_confidence: bool,
    min_data_points: int,
    weekly_rule: str = "W-MON",
    monthly_rule: str = "MS",
    non_negative: bool = True,
) -> PredictionPipeline:
    return PredictionPipeline(
        prediction_frequency=prediction_frequency,
        aggregation_level=aggregation_level,
        prediction_period=prediction_period,
        feature_columns=feature_columns or [],
        return_confidence=return_confidence,
        min_data_points=min_data_points,
        weekly_rule=weekly_rule,
        monthly_rule=monthly_rule,
        non_negative=non_negative,
    )
