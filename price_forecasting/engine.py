from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from prophet import Prophet  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    try:
        from fbprophet import Prophet  # type: ignore[import-not-found]
    except Exception:
        Prophet = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


HORIZONS = (7, 14, 30)


@dataclass
class ForecastResult:
    crop: str
    mandi: str
    forecast: Dict[int, float]
    lower_ci: Dict[int, float]
    upper_ci: Dict[int, float]
    model_weights: Dict[str, float]
    confidence: float


class CropPriceForecastingEngine:
    """
    Weighted ensemble engine for crop-price forecasting.

    Expected AGMARKNET-like input columns:
    - date, crop, mandi, price
    Optional columns:
    - rainfall_index, msp_floor
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

    def load_and_preprocess(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Price dataset not found: {self.csv_path}. "
                "Add AGMARKNET daily mandi prices CSV with columns date,crop,mandi,price."
            )

        df = pd.read_csv(self.csv_path)
        required = {"date", "crop", "mandi", "price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "crop", "mandi", "price"]).copy()
        df["crop"] = df["crop"].astype(str).str.strip()
        df["mandi"] = df["mandi"].astype(str).str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])

        agg_map = {"price": "mean"}
        if "rainfall_index" in df.columns:
            df["rainfall_index"] = pd.to_numeric(df["rainfall_index"], errors="coerce")
            agg_map["rainfall_index"] = "mean"
        if "msp_floor" in df.columns:
            df["msp_floor"] = pd.to_numeric(df["msp_floor"], errors="coerce")
            agg_map["msp_floor"] = "mean"

        df = df.groupby(["crop", "mandi", "date"], as_index=False).agg(agg_map)
        df = df.sort_values(["crop", "mandi", "date"])

        # Optional external features
        if "rainfall_index" not in df.columns:
            df["rainfall_index"] = 0.0
        if "msp_floor" not in df.columns:
            df["msp_floor"] = 0.0

        # Harvest calendar flags (simplified, crop-sensitive)
        df["month"] = df["date"].dt.month
        df["is_harvest_window"] = df.apply(self._harvest_flag, axis=1).astype(int)

        # Lag and rolling features
        for lag in (7, 14, 30):
            df[f"lag_{lag}"] = df.groupby(["crop", "mandi"])["price"].shift(lag)

        for w in (7, 14, 30):
            df[f"roll_mean_{w}"] = df.groupby(["crop", "mandi"])["price"].transform(
                lambda s: s.shift(1).rolling(w).mean()
            )
            df[f"roll_std_{w}"] = df.groupby(["crop", "mandi"])["price"].transform(
                lambda s: s.shift(1).rolling(w).std()
            )

        df["dow"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear

        # Use lag values to maintain continuity when nulls exist
        num_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
        df[num_cols] = df[num_cols].bfill().ffill()
        df = df.dropna(subset=["price"])

        self.raw_df = df.copy()
        self.df = df
        return df

    def _harvest_flag(self, row: pd.Series) -> int:
        crop = str(row["crop"]).lower()
        m = int(row["month"])

        # approximate windows
        if crop in {"wheat"}:
            return int(m in {3, 4, 5})
        if crop in {"rice", "paddy"}:
            return int(m in {10, 11, 12})
        if crop in {"maize"}:
            return int(m in {9, 10, 11})
        if crop in {"cotton"}:
            return int(m in {10, 11, 12, 1})
        if crop in {"tomato", "onion", "potato"}:
            return int(m in {1, 2, 3, 10, 11, 12})
        return 0

    def forecast_crop_mandi(self, crop: str, mandi: str) -> ForecastResult:
        if self.df is None:
            self.load_and_preprocess()

        assert self.df is not None
        series_df = self.df[(self.df["crop"].str.lower() == crop.lower()) & (self.df["mandi"].str.lower() == mandi.lower())].copy()
        if len(series_df) < 120:
            raise ValueError(
                f"Not enough history for {crop}-{mandi}. Need >=120 daily records, found {len(series_df)}."
            )

        series_df = series_df.sort_values("date").reset_index(drop=True)
        valid_span = min(45, max(14, len(series_df) // 5))
        train_df = series_df.iloc[:-valid_span].copy()
        valid_df = series_df.iloc[-valid_span:].copy()

        prophet_pred, prophet_sigma = self._fit_predict_prophet(train_df, valid_df)
        xgb_pred, xgb_sigma, xgb_model = self._fit_predict_xgboost(train_df, valid_df)
        lstm_pred, lstm_sigma = self._fit_predict_lstm(train_df, valid_df)

        y_valid = np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float)
        model_preds = {
            "prophet": prophet_pred,
            "xgboost": xgb_pred,
            "lstm": lstm_pred,
        }
        model_sigmas = {
            "prophet": prophet_sigma,
            "xgboost": xgb_sigma,
            "lstm": lstm_sigma,
        }

        weights = self._optimize_weights(y_valid, model_preds)

        # Final fit on full series and future predictions
        future_prophet = self._forecast_prophet_future(series_df)
        future_xgb = self._forecast_xgb_future(series_df, xgb_model)
        future_lstm = self._forecast_lstm_future(series_df)

        forecast = {}
        lower = {}
        upper = {}
        latest_price = float(series_df["price"].iloc[-1])

        for h in HORIZONS:
            p = (
                weights["prophet"] * future_prophet[h]
                + weights["xgboost"] * future_xgb[h]
                + weights["lstm"] * future_lstm[h]
            )
            sigma = (
                weights["prophet"] * model_sigmas["prophet"]
                + weights["xgboost"] * model_sigmas["xgboost"]
                + weights["lstm"] * model_sigmas["lstm"]
            )

            # Slightly wider intervals for longer horizon
            horizon_factor = 1.0 + (h / 30.0) * 0.4
            margin = 1.96 * sigma * horizon_factor
            forecast[h] = float(max(0.0, p))
            lower[h] = float(max(0.0, p - margin))
            upper[h] = float(max(0.0, p + margin))

        # Confidence score from relative interval width and agreement
        width_ratio = np.mean(
            [
                (upper[h] - lower[h]) / max(1e-6, forecast[h])
                for h in HORIZONS
            ]
        )
        agreement = np.mean(
            [
                np.std([future_prophet[h], future_xgb[h], future_lstm[h]]) / max(1e-6, latest_price)
                for h in HORIZONS
            ]
        )
        confidence = float(max(35.0, min(95.0, 92.0 - width_ratio * 40.0 - agreement * 120.0)))

        return ForecastResult(
            crop=crop,
            mandi=mandi,
            forecast=forecast,
            lower_ci=lower,
            upper_ci=upper,
            model_weights=weights,
            confidence=confidence,
        )

    def _feature_cols(self) -> List[str]:
        return [
            "lag_7",
            "lag_14",
            "lag_30",
            "roll_mean_7",
            "roll_std_7",
            "roll_mean_14",
            "roll_std_14",
            "roll_mean_30",
            "roll_std_30",
            "is_harvest_window",
            "msp_floor",
            "rainfall_index",
            "dow",
            "day_of_year",
        ]

    def _fit_predict_prophet(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        if Prophet is None:
            baseline = np.full(len(valid_df), train_df["price"].iloc[-1], dtype=float)
            sigma = float(np.std(np.asarray(train_df["price"].tail(30).to_numpy(dtype=float), dtype=float)))
            return baseline, sigma

        try:
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            p_train = train_df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
            m.fit(p_train)
            future = pd.DataFrame({"ds": valid_df["date"]})
            pred = np.asarray(m.predict(future)["yhat"].to_numpy(dtype=float), dtype=float)
            sigma = float(
                np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred)
            )
            return pred, max(1.0, sigma)
        except Exception:
            baseline = np.full(len(valid_df), float(train_df["price"].iloc[-1]), dtype=float)
            sigma = float(np.std(np.asarray(train_df["price"].tail(30).to_numpy(dtype=float), dtype=float)))
            return baseline, max(1.0, sigma)

    def _forecast_prophet_future(self, full_df: pd.DataFrame) -> Dict[int, float]:
        if Prophet is None:
            last = float(full_df["price"].iloc[-1])
            return {h: last for h in HORIZONS}

        try:
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            p_df = full_df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
            m.fit(p_df)
            future = m.make_future_dataframe(periods=max(HORIZONS), freq="D", include_history=False)
            pred = np.asarray(m.predict(future)["yhat"].to_numpy(dtype=float), dtype=float)
            return {h: float(pred[h - 1]) for h in HORIZONS}
        except Exception:
            last = float(full_df["price"].iloc[-1])
            return {h: last for h in HORIZONS}

    def _fit_predict_xgboost(
        self, train_df: pd.DataFrame, valid_df: pd.DataFrame
    ) -> Tuple[np.ndarray, float, Optional[Any]]:
        feats = self._feature_cols()
        x_train = train_df[feats].fillna(0.0)
        y_train = train_df["price"]
        x_valid = valid_df[feats].fillna(0.0)

        if xgb is None:
            pred = np.full(len(valid_df), float(y_train.iloc[-1]))
            sigma = float(np.std(valid_df["price"].values - pred))
            return pred, max(1.0, sigma), None

        try:
            model = xgb.XGBRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_valid), dtype=float)
            sigma = float(np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred))
            return pred, max(1.0, sigma), model
        except Exception:
            pred = np.full(len(valid_df), float(y_train.iloc[-1]))
            sigma = float(np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred))
            return pred, max(1.0, sigma), None

    def _forecast_xgb_future(self, full_df: pd.DataFrame, model: Optional[Any]) -> Dict[int, float]:
        if model is None:
            last = float(full_df["price"].iloc[-1])
            return {h: last for h in HORIZONS}

        feats = self._feature_cols()
        df = full_df.copy()
        forecasts = {}

        for step in range(1, max(HORIZONS) + 1):
            next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
            row = {
                "date": next_date,
                "price": np.nan,
                "rainfall_index": float(df["rainfall_index"].iloc[-7:].mean()),
                "msp_floor": float(df["msp_floor"].iloc[-1]),
                "month": int(next_date.month),
                "is_harvest_window": int(self._harvest_flag(pd.Series({"crop": df["crop"].iloc[-1], "month": next_date.month}))),
                "dow": int(next_date.dayofweek),
                "day_of_year": int(next_date.dayofyear),
            }

            temp = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            for lag in (7, 14, 30):
                temp.loc[temp.index[-1], f"lag_{lag}"] = temp["price"].shift(lag).iloc[-1]
            for w in (7, 14, 30):
                temp.loc[temp.index[-1], f"roll_mean_{w}"] = temp["price"].shift(1).rolling(w).mean().iloc[-1]
                temp.loc[temp.index[-1], f"roll_std_{w}"] = temp["price"].shift(1).rolling(w).std().iloc[-1]

            x_next = temp.iloc[[-1]][feats].ffill().bfill().fillna(0.0)
            try:
                pred = float(model.predict(x_next)[0])
            except Exception:
                pred = float(df["price"].iloc[-1])
            temp.loc[temp.index[-1], "price"] = pred
            df = temp
            if step in HORIZONS:
                forecasts[step] = pred

        return forecasts

    def _fit_predict_lstm(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        if torch is None or nn is None:
            pred = np.full(len(valid_df), float(train_df["price"].iloc[-1]))
            sigma = float(np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred))
            return pred, max(1.0, sigma)
        assert torch is not None and nn is not None
        nn_mod: Any = nn

        class _LSTMRegressor(nn.Module):
            def __init__(self, input_size: int = 1, hidden_size: int = 24):
                super().__init__()
                self.lstm = nn_mod.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.fc = nn_mod.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        series = np.asarray(train_df["price"].to_numpy(dtype=np.float32), dtype=np.float32)
        lookback = 30
        if len(series) <= lookback + 10:
            pred = np.full(len(valid_df), float(series[-1]))
            sigma = float(np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred))
            return pred, max(1.0, sigma)

        mean = float(series.mean())
        std = float(series.std()) or 1.0
        norm = (series - mean) / std

        xs, ys = [], []
        for i in range(lookback, len(norm)):
            xs.append(norm[i - lookback : i])
            ys.append(norm[i])

        x_t = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

        model = _LSTMRegressor()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(25):
            optimizer.zero_grad()
            out = model(x_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            optimizer.step()

        # validation recursive prediction
        model.eval()
        history = np.asarray(train_df["price"].to_numpy(dtype=np.float32), dtype=np.float32).tolist()
        preds = []
        for _ in range(len(valid_df)):
            seq = np.array(history[-lookback:], dtype=np.float32)
            seq_n = (seq - mean) / std
            x_in = torch.tensor(seq_n.reshape(1, lookback, 1), dtype=torch.float32)
            with torch.no_grad():
                y_hat_n = float(model(x_in).item())
            y_hat = y_hat_n * std + mean
            preds.append(y_hat)
            history.append(y_hat)

        pred_arr = np.array(preds)
        sigma = float(np.std(np.asarray(valid_df["price"].to_numpy(dtype=float), dtype=float) - pred_arr))
        return pred_arr, max(1.0, sigma)

    def _forecast_lstm_future(self, full_df: pd.DataFrame) -> Dict[int, float]:
        if torch is None or nn is None:
            last = float(full_df["price"].iloc[-1])
            return {h: last for h in HORIZONS}
        assert torch is not None and nn is not None
        nn_mod: Any = nn

        class _LSTMRegressor(nn.Module):
            def __init__(self, input_size: int = 1, hidden_size: int = 24):
                super().__init__()
                self.lstm = nn_mod.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.fc = nn_mod.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        series = np.asarray(full_df["price"].to_numpy(dtype=np.float32), dtype=np.float32)
        lookback = 30
        if len(series) <= lookback + 10:
            last = float(series[-1])
            return {h: last for h in HORIZONS}

        mean = float(series.mean())
        std = float(series.std()) or 1.0
        norm = (series - mean) / std

        xs, ys = [], []
        for i in range(lookback, len(norm)):
            xs.append(norm[i - lookback : i])
            ys.append(norm[i])

        x_t = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

        model = _LSTMRegressor()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            out = model(x_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        history = list(series)
        out = {}
        for step in range(1, max(HORIZONS) + 1):
            seq = np.array(history[-lookback:], dtype=np.float32)
            seq_n = (seq - mean) / std
            x_in = torch.tensor(seq_n.reshape(1, lookback, 1), dtype=torch.float32)
            with torch.no_grad():
                y_hat_n = float(model(x_in).item())
            y_hat = y_hat_n * std + mean
            history.append(y_hat)
            if step in HORIZONS:
                out[step] = float(y_hat)
        return out

    def _optimize_weights(self, y_true: np.ndarray, model_preds: Dict[str, np.ndarray]) -> Dict[str, float]:
        # Inverse MAE initialization
        maes = {}
        for name, pred in model_preds.items():
            maes[name] = max(1e-6, mean_absolute_error(y_true, pred))

        init = {name: 1.0 / mae for name, mae in maes.items()}
        s = sum(init.values())
        weights = {k: v / s for k, v in init.items()}

        # tiny CV search around inverse-MAE weights
        best_weights = weights.copy()
        best_loss = self._ensemble_loss(y_true, model_preds, best_weights)

        grid = np.linspace(0.1, 0.8, 8)
        for wp in grid:
            for wx in grid:
                wl = 1.0 - wp - wx
                if wl < 0.1 or wl > 0.8:
                    continue
                cand = {"prophet": float(wp), "xgboost": float(wx), "lstm": float(wl)}
                loss = self._ensemble_loss(y_true, model_preds, cand)
                if loss < best_loss:
                    best_loss = loss
                    best_weights = cand

        return best_weights

    @staticmethod
    def _ensemble_loss(y_true: np.ndarray, model_preds: Dict[str, np.ndarray], w: Dict[str, float]) -> float:
        y_hat = (
            w["prophet"] * model_preds["prophet"]
            + w["xgboost"] * model_preds["xgboost"]
            + w["lstm"] * model_preds["lstm"]
        )
        return float(mean_absolute_error(y_true, y_hat))
