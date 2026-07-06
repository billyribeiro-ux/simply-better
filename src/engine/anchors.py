"""Session anchors: VWAP, day open, prior close, prior-day POC, developing POC.

POC = point of control — the price bin with maximum traded volume. Histograms
are built incrementally from bar typical prices so the developing POC at bar i
uses volume from bars 0..i ONLY (no lookahead). Prior-day POC is the finished
histogram of the previous session, carried across the day loop by the caller.

Bin width is ATR-adaptive: max(tick, ATR/50), fixed at session start from the
prior day's ATR so bin edges never shift intraday.
"""
from __future__ import annotations

import numpy as np

ANCHORS: list[str] = ["vwap", "day_open", "prior_close", "pd_poc", "dev_poc"]

# anchor name -> the per-event room feature carrying its runway in ATR units
ROOM_FEATURE: dict[str, str] = {
    "vwap": "room_vwap_atr",
    "day_open": "room_open_atr",
    "prior_close": "room_pclose_atr",
    "pd_poc": "room_pdpoc_atr",
    "dev_poc": "room_dpoc_atr",
}


class PriceHistogram:
    """Incremental volume-at-price histogram keyed by integer bin index."""

    __slots__ = ("bin_w", "_vol")

    def __init__(self, atr: float, tick: float = 0.01) -> None:
        self.bin_w = max(tick, atr / 50.0)
        self._vol: dict[int, float] = {}

    def add_bar(self, high: float, low: float, close: float, volume: float) -> None:
        """Spread a bar's volume uniformly across the bins its range touched."""
        if volume <= 0:
            return
        b_lo = int(np.floor(low / self.bin_w))
        b_hi = int(np.floor(high / self.bin_w))
        n = b_hi - b_lo + 1
        per = volume / n
        for b in range(b_lo, b_hi + 1):
            self._vol[b] = self._vol.get(b, 0.0) + per

    def poc(self) -> float | None:
        if not self._vol:
            return None
        b = max(self._vol.items(), key=lambda kv: kv[1])[0]
        return (b + 0.5) * self.bin_w

    def is_empty(self) -> bool:
        return not self._vol


def anchor_prices(vwap: float, day_open: float, prior_close: float,
                  pd_poc: float | None, dev_poc: float | None) -> dict[str, float | None]:
    return {
        "vwap": vwap,
        "day_open": day_open,
        "prior_close": prior_close,
        "pd_poc": pd_poc,
        "dev_poc": dev_poc,
    }


def anchor_room_atr(anchor_px: float | None, ref_px: float, side: float,
                    atr: float) -> float:
    """Signed distance from ref price to anchor IN THE PROFIT DIRECTION, ATR units.

    side = +1 long / -1 short. Positive = anchor sits in the trade's profit
    direction (there is runway). Negative or None-anchor = no runway.
    """
    if anchor_px is None or atr <= 0:
        return 0.0
    return float(side * (anchor_px - ref_px) / atr)
