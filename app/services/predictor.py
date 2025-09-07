from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from prophet import Prophet

@dataclass
class EnsembleForecaster:
    frequency: str                 # 'weekly' | 'monthly'
    horizon: int
    return_confidence: bool = False
    weekly_rule: str = "W-MON"     # Pazartesi
    monthly_rule: str = "MS"       # Ay başı

    def _freq_rule(self) -> str:
        if self.frequency == "monthly":
            return self.monthly_rule
        # default weekly
        return self.weekly_rule

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        # Prophet baseline
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(df.rename(columns={"ds": "ds", "y": "y"}))

        # Gelecek veri çerçevesi: tek ankraj
        rule = self._freq_rule()
        future = model.make_future_dataframe(periods=self.horizon, freq=rule, include_history=False)
        forecast = model.predict(future)
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        out = forecast[cols].copy()

        # Basit backtest (son n nokta)
        metrics = {"mae": None, "rmse": None, "mape": None}
        n_val = max(0, min(max(4, self.horizon), max(2, len(df) // 3)))
        if len(df) >= n_val + 5 and n_val >= 2:
            train_df = df.iloc[:-n_val].copy()
            val_df = df.iloc[-n_val:].copy()

            bt_model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )
            bt_model.fit(train_df.rename(columns={"ds": "ds", "y": "y"}))

            bt_future = bt_model.make_future_dataframe(periods=n_val, freq=rule, include_history=False)
            bt_forecast = bt_model.predict(bt_future)[["ds", "yhat"]]

            merged = val_df.merge(bt_forecast, on="ds", how="inner")
            if len(merged) >= 2:
                y_true = merged["y"].astype(float)
                y_pred = merged["yhat"].astype(float)
                mae = (y_true - y_pred).abs().mean()
                rmse = (((y_true - y_pred) ** 2).mean()) ** 0.5
                non_zero = y_true != 0
                mape = float(((y_true[non_zero] - y_pred[non_zero]).abs() / y_true[non_zero]).mean() * 100) if non_zero.any() else None
                metrics = {"mae": float(mae), "rmse": float(rmse), "mape": mape}

        # Model ve veri özeti
        model_info = {
            # Gerçek ensemble yoksa etiket pipeline'da Prophet'e normalize edilecek
            "algorithm": "Ensemble (Prophet)",
            "accuracy_mae": metrics["mae"],
            "accuracy_rmse": metrics["rmse"],
            "accuracy_mape": metrics["mape"],
            "backtest_points": int(n_val),
        }
        data_summary = {
            "input_rows": int(len(df)),
            "date_range": f"{pd.to_datetime(df['ds'].min()).date()} to {pd.to_datetime(df['ds'].max()).date()}",
            "features_used": ["trend", "weekly_seasonality", "yearly_seasonality"],
        }

        if not self.return_confidence:
            out = out.drop(columns=["yhat_lower", "yhat_upper"])

        # Sadece ileri dönemleri döndürdüğümüzden emin ol (include_history=False zaten yeterli)
        return out.sort_values("ds").reset_index(drop=True), model_info, data_summary
