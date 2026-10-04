"""Data-layer models shared by the REST client and the price stream."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, computed_field

#: Columns of every candle DataFrame (index: UTC ``DatetimeIndex`` named ``time``).
CANDLE_COLUMNS = ["open", "high", "low", "close", "volume", "complete"]


class InstrumentInfo(BaseModel):
    name: str
    type: str | None = None
    display_name: str | None = None
    pip_location: int | None = None
    display_precision: int | None = None


class PriceTick(BaseModel):
    instrument: str
    time: datetime
    bid: float
    ask: float
    tradeable: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float:
        return self.ask - self.bid


def empty_candles() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "complete": pd.Series(dtype="bool"),
        },
        index=pd.DatetimeIndex([], tz="UTC", name="time"),
    )
    return frame


def completed(df: pd.DataFrame) -> pd.DataFrame:
    """Only finished candles: what strategies analyse. The chart keeps the live one."""
    if df.empty or "complete" not in df.columns:
        return df
    return df[df["complete"].astype(bool)]
