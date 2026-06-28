"""
OB_BoS_Test_Simple.py
Order Block with Eventual BOS (Break of Structure) Strategy Backtester
Mirrors SD_Test_Simple.py and OB_Test_Simple.py structure exactly.
────────────────────────────────────────────────────────────────────────
CONCEPT:
  Traditional OB: displacement candle ITSELF must break structure
  OB_BoS (Eventual BOS): find an opposite origin candle, then WAIT
  for price to eventually break a swing high/low — BOS can happen
  N bars later. Zone drawn at origin candle retroactively.

THREE-STEP LOGIC:
  Step 1 — ORIGIN: opposite candle with body >= ATR × min_body_mult
            with-trend filter: close > EMA (bull) / close < EMA (bear)

  Step 2 — WATCH: track BOS level (swing high/low from origin bar)
            Cancel if:
              - close beyond origin extreme (opposite side) → invalidated
              - waited > max_wait_bars → expired
            Confirm if:
              - close beyond BOS level → zone confirmed

  Step 3 — ZONE:
            If bar before origin is ALSO opposite direction → 2-bar zone
            Otherwise 1-bar zone
            Zone = high/low range of origin candle(s)

ENTRY MODES (iterated):
  top    : Buy Limit @ zone TOP    / Sell Limit @ zone BOTTOM (earliest)
  mid    : Buy Limit @ zone MID    / Sell Limit @ zone MID
  bottom : Buy Limit @ zone BOTTOM / Sell Limit @ zone TOP   (deepest)

SL PLACEMENT (anchored to zone extreme regardless of entry):
  BUY:  SL = zone_bottom - sl_buffer
  SELL: SL = zone_top    + sl_buffer

TP: entry ± (entry - SL) × RR

PARAMETERS ITERATED:
  BOS lookback : 3, 5, 10
  SL buffer    : 5, 10, 15 pips
  RR           : 1.5, 2.0, 2.5, 3.0
  Entry mode   : top, mid, bottom
  → 3 × 3 × 4 × 3 = 108 combos per pair/TF

FIXED:
  EMA length     : 100
  min_body_mult  : 0.5 × ATR
  max_wait_bars  : 20

USAGE:
  Single run:
    python OB_BoS_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv

  Entry matrix (top/mid/bottom × none/origination — 6 combos):
    python OB_BoS_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv --entry_matrix

  Full parameter matrix (108 combos):
    python OB_BoS_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv --matrix

  Multi pairs × TFs:
    python OB_BoS_Test_Simple.py --multi --entry bottom

OUTPUTS:
  OB_BoS_Results_<label>.csv
  OB_BoS_Matrix_<pair>_<tf>.csv
  OB_BoS_Entry_Comparison_<pair>_<tf>.csv
  OB_BoS_Multi_Summary.csv
"""

import argparse
import copy
import re
import pandas as pd
import numpy as np
from pathlib import Path
import pytz

# ── FIXED PARAMETERS ─────────────────────────────────────────────────────────
EMA_LENGTH      = 100
ATR_LENGTH      = 14
MIN_BODY_MULT   = 0.5      # origin candle min body size × ATR
MAX_WAIT_BARS   = 20       # max bars to wait for BOS
MAX_ZONES       = 100      # max pending + confirmed zones tracked
BRISBANE_TZ     = pytz.timezone("Australia/Brisbane")

# ── PROXIMITY FILTER — minimum pip separation between zones ──────────────────
# Prevents sandwich trades where zone 2 SL is hit before zone 1 fills
# Values in pips — scaled by pip_size at runtime
PROXIMITY_BY_TF = {
    1:  15,   # 1H  = 15 pips
    2:  20,   # 2H  = 20 pips
    4:  30,   # 4H  = 30 pips
    8:  40,   # 8H  = 40 pips
    24: 60,   # 1D  = 60 pips
}
ZONE_FILTER_MODES = ["none", "older", "stronger"]

# ── PARAMETER GRIDS ──────────────────────────────────────────────────────────
BOS_LOOKBACKS  = [3, 5, 10]
SL_BUFFERS     = [5, 10, 15]     # pips
RR_VALUES      = [1.5, 2.0, 2.5, 3.0]
ENTRY_MODES    = ["top", "mid", "bottom"]

# ── PAIRS AND TFs ─────────────────────────────────────────────────────────────
MULTI_PAIRS = [
    "GBPUSD","EURUSD","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "XAUUSD","USDJPY","AUDCAD","AUDNZD"
]
MULTI_TF_MINS   = [60, 120, 240, 480, 1440]
MULTI_TF_LABELS = {60:"1H", 120:"2H", 240:"4H", 480:"8H", 1440:"1D"}

# ── SL SCALING ────────────────────────────────────────────────────────────────
SL_MAX_BY_TF = {
    1:  (7,   20),
    2:  (10,  30),
    4:  (15,  45),
    8:  (20,  60),
    24: (30,  90),
}
SL_MAX_BY_TF_GOLD = {
    1:  (50,  150),
    2:  (80,  250),
    4:  (120, 400),
    8:  (200, 600),
    24: (300, 900),
}
# ─────────────────────────────────────────────────────────────────────────────


def detect_pair_tf(csv_name: str):
    name = Path(csv_name).stem.upper()
    tf_hours = None
    m = re.search(r'__(\d+)$', name)
    if m:
        mins     = int(m.group(1))
        tf_hours = mins // 60 if mins >= 60 else 1
    elif name.endswith('__D') or name.endswith('_D'):
        tf_hours = 24
    pair = None
    m2 = re.search(r'_([A-Z]{6})_', name)
    if m2:
        pair = m2.group(1)
    return pair, tf_hours


def get_pip_size(pair: str) -> float:
    if pair is None: return 0.0001
    p = pair.upper()
    if "JPY" in p: return 0.01
    if "XAU" in p or "GOLD" in p: return 0.10
    return 0.0001


def is_gold(pair: str) -> bool:
    return pair is not None and ("XAU" in pair.upper() or "GOLD" in pair.upper())


def get_sl_defaults(pair: str, tf_hours: int) -> tuple:
    if is_gold(pair):
        return SL_MAX_BY_TF_GOLD.get(tf_hours, (80, 250))
    return SL_MAX_BY_TF.get(tf_hours, (10, 30))


def parse_args():
    p = argparse.ArgumentParser(description="OB_BoS Strategy Backtest")
    p.add_argument("--csv",           default=None)
    p.add_argument("--pair",          default=None)
    p.add_argument("--tf_hours",      default=None,   type=int)
    p.add_argument("--pip_size",      default=None,   type=float)
    p.add_argument("--sl_buffer",     default=10,     type=int)
    p.add_argument("--max_sl",        default=None,   type=int)
    p.add_argument("--rr",            default=1.5,    type=float)
    p.add_argument("--bos_lookback",  default=5,      type=int)
    p.add_argument("--entry",         default="bottom",
                   choices=["top","mid","bottom"])
    p.add_argument("--trend_filter",  default="origination",
                   choices=["none","origination"])
    p.add_argument("--zone_filter",   default="none",
                   choices=ZONE_FILTER_MODES,
                   help="Zone conflict resolution: "
                        "none=keep all | "
                        "older=keep first confirmed zone when too close | "
                        "stronger=keep zone with higher BOS bar volume")
    p.add_argument("--entry_matrix",  action="store_true",
                   help="Run top/mid/bottom × none/origination (6 combos)")
    p.add_argument("--matrix",        action="store_true",
                   help="Run full parameter matrix (108 combos)")
    p.add_argument("--multi",         action="store_true",
                   help="Run all pairs × TFs")
    return p.parse_args()


def resolve_params(args, csv_name: str):
    pair, tf_hours = detect_pair_tf(csv_name)
    if args.pair:     pair     = args.pair.upper()
    if args.tf_hours: tf_hours = args.tf_hours
    if tf_hours is None: tf_hours = 2
    if pair     is None: pair     = "UNKNOWN"
    pip_size = args.pip_size if args.pip_size else get_pip_size(pair)
    default_sl_buf, default_max_sl = get_sl_defaults(pair, tf_hours)
    sl_buffer = args.sl_buffer if args.sl_buffer is not None else default_sl_buf
    max_sl    = args.max_sl    if args.max_sl    is not None else default_max_sl
    return pair, tf_hours, pip_size, sl_buffer, max_sl


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    if len(df.columns) < 4:
        df = pd.read_csv(csv_path, sep=",")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = (
        pd.to_datetime(df["time"], unit="s", utc=True)
        .dt.tz_convert(BRISBANE_TZ)
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    return df


def calc_atr(df, length=ATR_LENGTH):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def calc_ema(series, length=EMA_LENGTH):
    return series.ewm(span=length, min_periods=length, adjust=False).mean()


def detect_ob_bos_zones(df, bos_lookback=5, trend_filter="origination",
                        zone_filter="none", tf_hours=2, pip_size=0.0001):
    """
    Detect OB_BoS zones using eventual BOS logic.

    Returns list of confirmed zone dicts — each zone has:
      top, bottom, formed_bar (BOS bar), origin_bar, formed_time,
      origin_bullish (trend at origin), is_active, trade_taken

    Three steps per bar:
      1. Check for new origin candles → add to pending list
      2. Update all pending zones:
         - Invalidate if close beyond origin extreme
         - Expire if waited > MAX_WAIT_BARS
         - Confirm if close beyond BOS level → move to confirmed
      3. Confirmed zones returned for backtesting
    """
    atr       = calc_atr(df)
    trend_ema = calc_ema(df["close"])

    # Volume — use if available, fallback to 1
    vol_col = df["volume"].values if "volume" in df.columns else None

    # Pending zone structure
    # {origin_bar, zone_top, zone_bottom, bos_level, is_bull,
    #  bars_waited, origin_bullish}
    bull_pending = []
    bear_pending = []
    confirmed    = []

    check_origin = trend_filter in ("origination",)

    for i in range(max(bos_lookback + 1, ATR_LENGTH + 1), len(df)):
        o = df["open"][i];  c = df["close"][i]
        h = df["high"][i];  l = df["low"][i]
        atr_v   = atr[i]
        ema_v   = trend_ema[i]
        if pd.isna(atr_v) or pd.isna(ema_v):
            continue

        body = abs(c - o)
        is_bull_trend = c > ema_v
        is_bear_trend = c < ema_v

        # ── STEP 1: detect new origin candles ────────────────────────
        # BOS level = highest CLOSE / lowest CLOSE of bos_lookback bars
        # BEFORE origin — using closes not wicks for structural integrity
        # A real BOS requires a bar to CLOSE beyond previous structure
        swing_high = max(df["close"][i-k] for k in range(1, bos_lookback+1))
        swing_low  = min(df["close"][i-k] for k in range(1, bos_lookback+1))

        # Bearish origin → potential bull zone
        # No trend filter here — checked at BOS confirmation bar
        if (c < o) and (body >= atr_v * MIN_BODY_MULT):
            prev_bear = df["close"][i-1] < df["open"][i-1]
            if prev_bear:
                z_top = max(h, df["high"][i-1])
                z_bot = min(l, df["low"][i-1])
                z_left = i - 1
            else:
                z_top = h; z_bot = l; z_left = i

            bull_pending.append({
                "origin_bar":     i,
                "left_bar":       z_left,
                "zone_top":       z_top,
                "zone_bottom":    z_bot,
                "bos_level":      swing_high,
                "bars_waited":    0,
                "origin_bullish": is_bull_trend,
                "formed_time":    df["datetime"][i],
            })
            if len(bull_pending) > MAX_ZONES:
                bull_pending.pop(0)

        # Bullish origin → potential bear zone
        # No trend filter here — checked at BOS confirmation bar
        if (c > o) and (body >= atr_v * MIN_BODY_MULT):
            prev_bull = df["close"][i-1] > df["open"][i-1]
            if prev_bull:
                z_top = max(h, df["high"][i-1])
                z_bot = min(l, df["low"][i-1])
                z_left = i - 1
            else:
                z_top = h; z_bot = l; z_left = i

            bear_pending.append({
                "origin_bar":     i,
                "left_bar":       z_left,
                "zone_top":       z_top,
                "zone_bottom":    z_bot,
                "bos_level":      swing_low,
                "bars_waited":    0,
                "origin_bullish": is_bull_trend,
                "formed_time":    df["datetime"][i],
            })
            if len(bear_pending) > MAX_ZONES:
                bear_pending.pop(0)

        # ── STEP 2 & 3: update pending, confirm or expire ────────────
        # Bull pending — watching for close above bos_level
        to_remove = []
        for idx, p in enumerate(bull_pending):
            if p["origin_bar"] >= i:
                continue
            p["bars_waited"] += 1

            invalidated = c < p["zone_bottom"]
            expired     = p["bars_waited"] > MAX_WAIT_BARS
            confirmed_  = c > p["bos_level"]

            # Trend filter checked at BOS confirmation bar not origin
            trend_ok_bull = (not check_origin) or is_bull_trend

            if invalidated or expired:
                to_remove.append(idx)
            elif confirmed_ and not trend_ok_bull:
                # BOS happened but wrong trend — expire the zone
                to_remove.append(idx)
            elif confirmed_ and trend_ok_bull:
                confirmed.append({
                    "type":           "bull_bos",
                    "zone_top":       p["zone_top"],
                    "zone_bottom":    p["zone_bottom"],
                    "left_bar":       p["left_bar"],
                    "formed_bar":     i,
                    "origin_bar":     p["origin_bar"],
                    "formed_time":    df["datetime"][i],
                    "origin_time":    p["formed_time"],
                    "origin_bullish": p["origin_bullish"],
                    "bos_body":       abs(c - df["open"][i]),
                    "is_active":      True,
                    "trade_taken":    False,
                })
                to_remove.append(idx)
        for idx in reversed(to_remove):
            bull_pending.pop(idx)

        # Bear pending — watching for close below bos_level
        to_remove = []
        for idx, p in enumerate(bear_pending):
            if p["origin_bar"] >= i:
                continue
            p["bars_waited"] += 1

            invalidated = c > p["zone_top"]
            expired     = p["bars_waited"] > MAX_WAIT_BARS
            confirmed_  = c < p["bos_level"]

            # Trend filter checked at BOS confirmation bar not origin
            trend_ok_bear = (not check_origin) or is_bear_trend

            if invalidated or expired:
                to_remove.append(idx)
            elif confirmed_ and not trend_ok_bear:
                # BOS happened but wrong trend — expire the zone
                to_remove.append(idx)
            elif confirmed_ and trend_ok_bear:
                confirmed.append({
                    "type":           "bear_bos",
                    "zone_top":       p["zone_top"],
                    "zone_bottom":    p["zone_bottom"],
                    "left_bar":       p["left_bar"],
                    "formed_bar":     i,
                    "origin_bar":     p["origin_bar"],
                    "formed_time":    df["datetime"][i],
                    "origin_time":    p["formed_time"],
                    "origin_bullish": p["origin_bullish"],
                    "bos_body":       abs(c - df["open"][i]),
                    "is_active":      True,
                    "trade_taken":    False,
                })
                to_remove.append(idx)
        for idx in reversed(to_remove):
            bear_pending.pop(idx)

    bull_zones = [z for z in confirmed if z["type"] == "bull_bos"]
    bear_zones = [z for z in confirmed if z["type"] == "bear_bos"]

    # Apply proximity filter if requested
    if zone_filter != "none":
        prox_pips = PROXIMITY_BY_TF.get(tf_hours, 20)
        prox_dist = prox_pips * pip_size
        bull_zones = _apply_proximity_filter(bull_zones, prox_dist, zone_filter)
        bear_zones = _apply_proximity_filter(bear_zones, prox_dist, zone_filter)

    return bull_zones, bear_zones, trend_ema


def _apply_proximity_filter(zones, prox_dist, mode):
    """
    Filter out zones that are too close to existing active zones.

    Two zones are 'too close' if the distance between their closest
    edges is less than prox_dist:
      distance = max(z1.bottom, z2.bottom) - min(z1.top, z2.top)
      if negative → zones overlap
      if < prox_dist → too close

    mode = 'older'    : keep the zone with earlier formed_bar
    mode = 'stronger' : keep the zone with larger BOS bar body size
                        (BOS bar = the impulsive bar that broke structure)
                        bigger body = stronger institutional confirmation

    Zones are processed in chronological order (formed_bar).
    Each new zone is checked against all already-accepted zones.
    If conflict → apply mode rule.
    """
    if not zones:
        return zones

    sorted_z = sorted(zones, key=lambda z: z["formed_bar"])
    accepted  = []

    for z in sorted_z:
        conflict = None
        for a in accepted:
            dist = max(z["zone_bottom"], a["zone_bottom"]) - \
                   min(z["zone_top"],    a["zone_top"])
            if dist < prox_dist:
                conflict = a
                break

        if conflict is None:
            accepted.append(z)
        else:
            if mode == "older":
                # Keep already-accepted (older) zone — skip new one
                pass
            elif mode == "stronger":
                # Keep zone with larger BOS bar body size
                # BOS bar body = abs(bos_close - bos_open) at confirmation bar
                new_bos  = z.get("bos_body", 0.0)
                prev_bos = conflict.get("bos_body", 0.0)
                if new_bos > prev_bos:
                    # New zone has stronger BOS — replace existing
                    accepted.remove(conflict)
                    accepted.append(z)
                # else keep existing — skip new one

    return accepted


def run_backtest(df, bull_zones, bear_zones, trend_ema,
                 trend_filter, rr, pip_size, sl_buffer_pips, max_sl_pips,
                 tf_hours=2, entry_mode="bottom"):
    """
    Bar-by-bar OB_BoS backtest.

    Entry modes:
      top    : BUY @ zone TOP    / SELL @ zone BOTTOM (earliest)
      mid    : BUY @ zone MID    / SELL @ zone MID
      bottom : BUY @ zone BOTTOM / SELL @ zone TOP    (deepest)

    SL always anchored to zone extreme:
      BUY:  SL = zone_bottom - sl_buffer
      SELL: SL = zone_top    + sl_buffer

    Invalidation:
      BUY zone:  close < zone_bottom → dead
      SELL zone: close > zone_top    → dead

    One trade per zone — trade_taken = True permanently after fill.
    """
    b_zones = copy.deepcopy(bull_zones)
    s_zones = copy.deepcopy(bear_zones)
    trades  = []

    sl_buf   = sl_buffer_pips * pip_size
    max_sl   = max_sl_pips    * pip_size

    check_at_entry = trend_filter in ("entry", "both")

    for i in range(len(df)):
        bar_high  = df["high"][i]
        bar_low   = df["low"][i]
        bar_close = df["close"][i]
        bar_time  = df["datetime"][i]
        ema_now   = trend_ema[i]
        is_bull_now = (bar_close > ema_now) if not pd.isna(ema_now) else True
        is_bear_now = (bar_close < ema_now) if not pd.isna(ema_now) else True

        # ── BULL ZONE → BUY ──────────────────────────────────────────
        for zone in b_zones:
            # Zone only active after BOS confirmation bar
            if not zone["is_active"] or zone["formed_bar"] >= i:
                continue

            ztop = zone["zone_top"]
            zbot = zone["zone_bottom"]
            mid  = (ztop + zbot) / 2.0
            iw   = zone.get("origin_bullish", True)

            # Invalidation: close below zone bottom
            if bar_close < zbot:
                zone["is_active"] = False
                zone.pop("pending", None)
                continue

            # SL always anchored to zone bottom
            sl    = zbot - sl_buf
            rdist_top = ztop - sl
            rdist_mid = mid  - sl
            rdist_bot = zbot - sl

            if entry_mode == "top":
                if not zone["trade_taken"] and bar_low <= ztop:
                    if rdist_top <= 0 or rdist_top > max_sl:
                        zone["trade_taken"] = True; continue
                    if check_at_entry and not is_bull_now:
                        zone["trade_taken"] = True; continue
                    tp = ztop + rdist_top * rr
                    trades.append(_make_trade(
                        bar_time,"BUY","Bull_BoS",ztop,sl,tp,
                        rdist_top,iw,ztop,zbot,
                        i-zone["formed_bar"],zone["origin_time"],
                        pip_size,entry_mode,is_bull_now))
                    zone["trade_taken"] = True

            elif entry_mode == "bottom":
                if not zone["trade_taken"] and bar_low <= zbot:
                    if rdist_bot <= 0 or rdist_bot > max_sl:
                        zone["trade_taken"] = True; continue
                    if check_at_entry and not is_bull_now:
                        zone["trade_taken"] = True; continue
                    tp = zbot + rdist_bot * rr
                    trades.append(_make_trade(
                        bar_time,"BUY","Bull_BoS",zbot,sl,tp,
                        rdist_bot,iw,ztop,zbot,
                        i-zone["formed_bar"],zone["origin_time"],
                        pip_size,entry_mode,is_bull_now))
                    zone["trade_taken"] = True

            else:  # mid
                if not zone["trade_taken"] and not zone.get("pending", False):
                    if bar_low <= ztop:
                        if rdist_mid <= 0 or rdist_mid > max_sl:
                            zone["trade_taken"] = True; continue
                        if check_at_entry and not is_bull_now:
                            zone["trade_taken"] = True; continue
                        zone["pending"] = True
                        zone["pending_mid"] = mid
                        zone["pending_sl"]  = sl
                        zone["pending_rd"]  = rdist_mid
                        zone["pending_iw"]  = iw
                if not zone["trade_taken"] and zone.get("pending", False):
                    if bar_low <= zone["pending_mid"]:
                        mf = zone["pending_mid"]
                        rd = zone["pending_rd"]
                        tp = mf + rd * rr
                        trades.append(_make_trade(
                            bar_time,"BUY","Bull_BoS",mf,
                            zone["pending_sl"],tp,rd,
                            zone["pending_iw"],ztop,zbot,
                            i-zone["formed_bar"],zone["origin_time"],
                            pip_size,entry_mode,is_bull_now))
                        zone["trade_taken"] = True
                        zone["pending"]     = False

        # ── BEAR ZONE → SELL ─────────────────────────────────────────
        for zone in s_zones:
            if not zone["is_active"] or zone["formed_bar"] >= i:
                continue

            ztop = zone["zone_top"]
            zbot = zone["zone_bottom"]
            mid  = (ztop + zbot) / 2.0
            iw   = not zone.get("origin_bullish", True)  # bear zone: with-trend if bearish

            # Invalidation: close above zone top
            if bar_close > ztop:
                zone["is_active"] = False
                zone.pop("pending", None)
                continue

            sl    = ztop + sl_buf
            rdist_top = sl - zbot
            rdist_mid = sl - mid
            rdist_bot = sl - ztop

            if entry_mode == "top":
                if not zone["trade_taken"] and bar_high >= zbot:
                    if rdist_top <= 0 or rdist_top > max_sl:
                        zone["trade_taken"] = True; continue
                    if check_at_entry and not is_bear_now:
                        zone["trade_taken"] = True; continue
                    tp = zbot - rdist_top * rr
                    trades.append(_make_trade(
                        bar_time,"SELL","Bear_BoS",zbot,sl,tp,
                        rdist_top,iw,ztop,zbot,
                        i-zone["formed_bar"],zone["origin_time"],
                        pip_size,entry_mode,is_bear_now))
                    zone["trade_taken"] = True

            elif entry_mode == "bottom":
                if not zone["trade_taken"] and bar_high >= ztop:
                    if rdist_bot <= 0 or rdist_bot > max_sl:
                        zone["trade_taken"] = True; continue
                    if check_at_entry and not is_bear_now:
                        zone["trade_taken"] = True; continue
                    tp = ztop - rdist_bot * rr
                    trades.append(_make_trade(
                        bar_time,"SELL","Bear_BoS",ztop,sl,tp,
                        rdist_bot,iw,ztop,zbot,
                        i-zone["formed_bar"],zone["origin_time"],
                        pip_size,entry_mode,is_bear_now))
                    zone["trade_taken"] = True

            else:  # mid
                if not zone["trade_taken"] and not zone.get("pending", False):
                    if bar_high >= zbot:
                        if rdist_mid <= 0 or rdist_mid > max_sl:
                            zone["trade_taken"] = True; continue
                        if check_at_entry and not is_bear_now:
                            zone["trade_taken"] = True; continue
                        zone["pending"] = True
                        zone["pending_mid"] = mid
                        zone["pending_sl"]  = sl
                        zone["pending_rd"]  = rdist_mid
                        zone["pending_iw"]  = iw
                if not zone["trade_taken"] and zone.get("pending", False):
                    if bar_high >= zone["pending_mid"]:
                        mf = zone["pending_mid"]
                        rd = zone["pending_rd"]
                        tp = mf - rd * rr
                        trades.append(_make_trade(
                            bar_time,"SELL","Bear_BoS",mf,
                            zone["pending_sl"],tp,rd,
                            zone["pending_iw"],ztop,zbot,
                            i-zone["formed_bar"],zone["origin_time"],
                            pip_size,entry_mode,is_bear_now))
                        zone["trade_taken"] = True
                        zone["pending"]     = False

        # ── RESOLVE OPEN TRADES ──────────────────────────────────────
        for t in trades:
            if t["result"] is not None: continue
            sl, tp = t["_sl"], t["_tp"]
            ps     = t["_ps"]
            if t["_dir"] == "buy":
                if bar_low  <= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,5)
                    t["exit_time"]=bar_time
                    t["pnl_pips"]=round((sl-t["entry_price"])/ps,1)
                elif bar_high >= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,5)
                    t["exit_time"]=bar_time
                    t["pnl_pips"]=round((tp-t["entry_price"])/ps,1)
            else:
                if bar_high >= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,5)
                    t["exit_time"]=bar_time
                    t["pnl_pips"]=round((t["entry_price"]-sl)/ps,1)
                elif bar_low  <= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,5)
                    t["exit_time"]=bar_time
                    t["pnl_pips"]=round((t["entry_price"]-tp)/ps,1)

    return trades


def _make_trade(bar_time, direction, zone_type, entry, sl, tp,
                rdist, iw, ztop, zbot, zone_age,
                origin_time, pip_size, entry_mode, is_with_trend):
    return {
        "entry_time":      bar_time,
        "type":            direction,
        "zone_type":       zone_type,
        "entry_mode":      entry_mode,
        "trend_at_origin": "with" if iw else "counter",
        "trend_at_entry":  "with" if is_with_trend else "counter",
        "zone_top":        round(ztop,  5),
        "zone_bottom":     round(zbot,  5),
        "entry_price":     round(entry, 5),
        "sl_price":        round(sl,    5),
        "tp_price":        round(tp,    5),
        "rr_applied":      round(rdist > 0 and tp != sl and abs(tp-entry)/rdist or 0, 2),
        "risk_pips":       round(rdist / pip_size, 1),
        "zone_age_bars":   zone_age,
        "origin_time":     origin_time,
        "result":          None,
        "exit_price":      None,
        "exit_time":       None,
        "pnl_pips":        None,
        "_sl": sl, "_tp": tp,
        "_dir": "buy" if direction == "BUY" else "sell",
        "_ps":  pip_size,
    }


def calc_stats(trades, label, rr, min_age=0,
               pair="", tf_hours=2, sl_buffer=10, max_sl=30,
               entry_mode="bottom", bos_lookback=5):
    clean  = [{k:v for k,v in t.items() if not k.startswith("_")} for t in trades]
    df_t   = pd.DataFrame(clean) if clean else pd.DataFrame()

    base = {
        "label": label, "pair": pair, "tf_hours": tf_hours,
        "entry_mode": entry_mode, "sl_buffer_pips": sl_buffer,
        "max_sl_pips": max_sl, "rr": rr, "bos_lookback": bos_lookback,
        "trades": 0,
    }
    if df_t.empty: return base

    closed = df_t[df_t["result"].isin(["WIN","LOSS"])]
    wins   = closed[closed["result"]=="WIN"]
    losses = closed[closed["result"]=="LOSS"]
    n      = len(closed)
    if n == 0: return base

    wr      = len(wins)/n*100
    gross_w = wins["pnl_pips"].sum()       if len(wins)   else 0
    gross_l = abs(losses["pnl_pips"].sum()) if len(losses) else 0
    pf      = gross_w/gross_l             if gross_l > 0 else 999.0
    total_p = gross_w - gross_l
    exp     = total_p / n

    # With-trend vs counter-trend split
    wt = closed[closed["trend_at_origin"]=="with"]
    ct = closed[closed["trend_at_origin"]=="counter"]
    wt_w = wt[wt["result"]=="WIN"]; wt_l = wt[wt["result"]=="LOSS"]
    ct_w = ct[ct["result"]=="WIN"]; ct_l = ct[ct["result"]=="LOSS"]
    wt_wr = len(wt_w)/len(wt)*100 if len(wt) else 0
    ct_wr = len(ct_w)/len(ct)*100 if len(ct) else 0
    wt_pf = wt_w["pnl_pips"].sum()/abs(wt_l["pnl_pips"].sum()) if len(wt_l)>0 else 999
    ct_pf = ct_w["pnl_pips"].sum()/abs(ct_l["pnl_pips"].sum()) if len(ct_l)>0 else 999

    # Equity simulation
    account = 10000.0; equity = 10000.0; peak = 10000.0
    max_dd_d = max_dd_p = 0.0
    for _, row in closed.sort_values("entry_time").iterrows():
        risk_d  = equity * 0.01
        equity += row["pnl_pips"] * (risk_d / row["risk_pips"])
        if equity > peak: peak = equity
        dd = peak - equity; ddp = dd/peak*100
        if dd > max_dd_d: max_dd_d = dd; max_dd_p = ddp

    return {**base,
        "trades":            n,
        "wins":              len(wins),
        "losses":            len(losses),
        "win_rate_%":        round(wr,      1),
        "profit_factor":     round(pf,      2),
        "total_pips":        round(total_p, 1),
        "exp_pips/trade":    round(exp,     1),
        "wt_trades":         len(wt),   "wt_wins": len(wt_w),
        "wt_wr_%":           round(wt_wr, 1), "wt_pf": round(wt_pf, 2),
        "ct_trades":         len(ct),   "ct_wins": len(ct_w),
        "ct_wr_%":           round(ct_wr, 1), "ct_pf": round(ct_pf, 2),
        "final_equity_$":    round(equity,   2),
        "net_return_%":      round((equity-account)/account*100, 1),
        "max_dd_$":          round(max_dd_d, 2),
        "max_dd_%":          round(max_dd_p, 1),
        "_trades_list":      trades,
    }


def save_trade_csv(trades, output_dir, label):
    clean = [{k:v for k,v in t.items() if not k.startswith("_")} for t in trades]
    if clean:
        pd.DataFrame(clean).to_csv(
            output_dir / f"OB_BoS_Results_{label}.csv", index=False)


def print_summary(s, tf_label=""):
    em = s.get("entry_mode","bottom").upper()
    print(f"\n{'='*62}")
    print(f"  ▶ OB_BoS  [{s['pair']} {tf_label}]  entry={em}")
    print(f"{'='*62}")
    print(f"  bos_lb={s['bos_lookback']}  sl={s['sl_buffer_pips']}p  "
          f"rr={s['rr']}  ema={EMA_LENGTH}  body={MIN_BODY_MULT}  "
          f"max_wait={MAX_WAIT_BARS}")
    print(f"{'='*62}")
    print(f"  Trades        : {s['trades']}  "
          f"({s.get('wins',0)}W / {s.get('losses',0)}L)")
    print(f"  Win rate      : {s.get('win_rate_%',0)}%")
    print(f"  Profit factor : {s.get('profit_factor',0)}")
    print(f"  Total pips    : {s.get('total_pips',0)}")
    print(f"  Net return    : {s.get('net_return_%',0)}%  "
          f"(${s.get('final_equity_$',10000):,.2f})")
    print(f"  Max DD        : {s.get('max_dd_%',0)}%")
    print(f"  ── With-trend : {s.get('wt_trades',0)}t  "
          f"WR {s.get('wt_wr_%',0)}%  PF {s.get('wt_pf',0)}")
    print(f"  ── Counter    : {s.get('ct_trades',0)}t  "
          f"WR {s.get('ct_wr_%',0)}%  PF {s.get('ct_pf',0)}")
    print(f"{'='*62}")


def main():
    args        = parse_args()
    trading_dir = Path(r"C:\Trading")

    # ── MULTI ──────────────────────────────────────────────────────────────
    if args.multi:
        print(f"\nOB_BoS MULTI MODE — {len(MULTI_PAIRS)} pairs × TFs")
        print(f"  Entry={args.entry}  BOS_lb={args.bos_lookback}  "
              f"RR={args.rr}  SL={args.sl_buffer}p\n")
        hdr = "─"*100
        print(hdr)
        print(f"  {'Pair':<8}{'TF':<5}{'Trades':>7}{'WR%':>6}{'PF':>6}"
              f"{'Pips':>8}{'Ret%':>6}{'MaxDD%':>7}{'Ret/DD':>7}"
              f"{'BullZ':>7}{'BearZ':>7}")
        print(hdr)
        all_stats = []; not_found = []
        for pair in MULTI_PAIRS:
            for tf_mins in [60, 120, 240, 480]:
                tf_label = MULTI_TF_LABELS[tf_mins]
                csv_path = trading_dir / f"PEPPERSTONE_{pair}__{tf_mins}.csv"
                if not csv_path.exists():
                    not_found.append(csv_path.name); continue
                tf_hours = tf_mins // 60
                sl_buf, max_sl = get_sl_defaults(pair, tf_hours)
                pip_size = get_pip_size(pair)
                try:
                    df = load_data(csv_path)
                    bz, sz, ema = detect_ob_bos_zones(
                        df, args.bos_lookback, args.trend_filter,
                        zone_filter=args.zone_filter,
                        tf_hours=tf_hours, pip_size=pip_size)
                    trades = run_backtest(
                        df, bz, sz, ema, args.trend_filter, args.rr,
                        pip_size, sl_buf, max_sl,
                        tf_hours=tf_hours, entry_mode=args.entry)
                    label = (f"{pair}_{tf_label}_lb{args.bos_lookback}"
                             f"_sl{sl_buf}_rr{str(args.rr).replace('.','')}"
                             f"_{args.entry}")
                    s = calc_stats(trades, label, args.rr, pair=pair,
                                   tf_hours=tf_hours, sl_buffer=sl_buf,
                                   max_sl=max_sl, entry_mode=args.entry,
                                   bos_lookback=args.bos_lookback)
                    all_stats.append(s)
                    save_trade_csv(trades, trading_dir, label)
                    rdd = round(s["net_return_%"]/s["max_dd_%"],2) \
                          if s.get("max_dd_%",0)>0 else 999
                    print(f"  {pair:<8}{tf_label:<5}{s['trades']:>7}"
                          f"{s.get('win_rate_%',0):>6.1f}"
                          f"{s.get('profit_factor',0):>6.2f}"
                          f"{s.get('total_pips',0):>8.1f}"
                          f"{s.get('net_return_%',0):>6.1f}"
                          f"{s.get('max_dd_%',0):>7.1f}{rdd:>7.2f}"
                          f"{len(bz):>7}{len(sz):>7}")
                except Exception as e:
                    print(f"  {pair:<8}{tf_label:<5}  ERROR: {e}")
        print(hdr)
        if all_stats:
            df_m = pd.DataFrame([{k:v for k,v in s.items()
                                   if not k.startswith("_")}
                                  for s in all_stats])
            df_m.to_csv(trading_dir/"OB_BoS_Multi_Summary.csv", index=False)
        return

    # ── ENTRY MATRIX ───────────────────────────────────────────────────────
    if args.entry_matrix:
        if not args.csv: print("ERROR: --csv required"); return
        csv_path = trading_dir / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        pair, tf_hours, pip_size, sl_buf, max_sl = resolve_params(args, args.csv)
        tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")
        combos   = [(em, tf) for em in ENTRY_MODES
                    for tf in ("origination","none")]

        print(f"\nOB_BoS ENTRY MATRIX  [{pair} {tf_label}]  —  6 combos")
        print(f"  bos_lb={args.bos_lookback}  sl={sl_buf}p  "
              f"max_sl={max_sl}p  RR={args.rr}  zone_filter={args.zone_filter}\n")

        df = load_data(csv_path)
        print(f"  Bars: {len(df)}  |  "
              f"{df['datetime'].iloc[0].date()} → "
              f"{df['datetime'].iloc[-1].date()}")

        zone_cache = {}
        for tf_filt in ("origination","none"):
            bz, sz, ema = detect_ob_bos_zones(
                df, args.bos_lookback, tf_filt,
                zone_filter=args.zone_filter,
                tf_hours=tf_hours, pip_size=pip_size)
            zone_cache[tf_filt] = (bz, sz, ema)
            print(f"  Zones [{tf_filt}]: Bull={len(bz)}  Bear={len(sz)}")

        hdr = "─"*100
        print(f"\n{hdr}")
        print(f"  {'Entry':<8}{'Filter':<14}{'Trades':>7}{'WR%':>6}{'PF':>6}"
              f"{'Pips':>8}{'Ret%':>6}{'MaxDD%':>7}{'Ret/DD':>7}"
              f"{'WT WR%':>8}{'CT WR%':>8}")
        print(hdr)

        all_stats = []; prev_em = None
        for em, tf_filt in combos:
            if em != prev_em and prev_em is not None: print()
            prev_em = em
            bz, sz, ema = zone_cache[tf_filt]
            label  = (f"{pair}_{tf_label}_{tf_filt}_{em}"
                      f"_lb{args.bos_lookback}"
                      f"_sl{sl_buf}_rr{str(args.rr).replace('.','')}")
            trades = run_backtest(df, bz, sz, ema, tf_filt, args.rr,
                                  pip_size, sl_buf, max_sl,
                                  tf_hours=tf_hours, entry_mode=em)
            s = calc_stats(trades, label, args.rr, pair=pair,
                           tf_hours=tf_hours, sl_buffer=sl_buf,
                           max_sl=max_sl, entry_mode=em,
                           bos_lookback=args.bos_lookback)
            all_stats.append(s)
            rdd = round(s["net_return_%"]/s["max_dd_%"],2) \
                  if s.get("max_dd_%",0)>0 else 999
            print(f"  {em:<8}{tf_filt:<14}{s['trades']:>7}"
                  f"{s.get('win_rate_%',0):>6.1f}"
                  f"{s.get('profit_factor',0):>6.2f}"
                  f"{s.get('total_pips',0):>8.1f}"
                  f"{s.get('net_return_%',0):>6.1f}"
                  f"{s.get('max_dd_%',0):>7.1f}{rdd:>7.2f}"
                  f"{s.get('wt_wr_%',0):>8.1f}"
                  f"{s.get('ct_wr_%',0):>8.1f}")
            save_trade_csv(trades, trading_dir, label)

        print(hdr)
        df_m = pd.DataFrame([{k:v for k,v in s.items()
                               if not k.startswith("_")}
                              for s in all_stats])
        best_ret = df_m.loc[df_m["net_return_%"].idxmax()]
        best_pf  = df_m.loc[df_m["profit_factor"].idxmax()]
        rdd_col  = df_m["net_return_%"]/df_m["max_dd_%"].clip(lower=0.1)
        best_rdd = df_m.loc[rdd_col.idxmax()]
        print(f"\n  ── Best combos ──")
        print(f"  Best return  : {best_ret['label']}  "
              f"{best_ret['net_return_%']}%")
        print(f"  Best PF      : {best_pf['label']}  "
              f"PF={best_pf['profit_factor']}")
        print(f"  Best Ret/DD  : {best_rdd['label']}")
        df_m.to_csv(
            trading_dir/f"OB_BoS_Entry_Comparison_{pair}_{tf_label}.csv",
            index=False)
        return

    # ── FULL PARAMETER MATRIX ──────────────────────────────────────────────
    if args.matrix:
        if not args.csv: print("ERROR: --csv required"); return
        csv_path = trading_dir / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        pair, tf_hours, pip_size, sl_buf_def, max_sl = \
            resolve_params(args, args.csv)
        tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")

        combos = [(lb, sl, rr, em)
                  for lb  in BOS_LOOKBACKS
                  for sl  in SL_BUFFERS
                  for rr  in RR_VALUES
                  for em  in ENTRY_MODES]

        print(f"\nOB_BoS FULL MATRIX  [{pair} {tf_label}]  "
              f"—  {len(combos)} combos")
        df = load_data(csv_path)
        print(f"  Bars: {len(df)}  |  "
              f"{df['datetime'].iloc[0].date()} → "
              f"{df['datetime'].iloc[-1].date()}\n")

        # Pre-compute zones per lookback × filter combo
        zone_cache = {}
        for lb in BOS_LOOKBACKS:
            for tf_filt in ("origination","none"):
                bz, sz, ema = detect_ob_bos_zones(
                    df, lb, tf_filt,
                    zone_filter=args.zone_filter,
                    tf_hours=tf_hours, pip_size=pip_size)
                zone_cache[(lb, tf_filt)] = (bz, sz, ema)
                print(f"  Zones [lb={lb} {tf_filt}]: "
                      f"Bull={len(bz)}  Bear={len(sz)}")

        hdr = "─"*100
        print(f"\n{hdr}")
        print(f"  {'LB':<4}{'SL':>4}{'RR':>5}{'Entry':<8}"
              f"{'Trades':>7}{'WR%':>6}{'PF':>6}"
              f"{'Ret%':>6}{'MaxDD%':>7}{'Ret/DD':>7}")
        print(hdr)

        all_stats = []; prev_lb = None
        for lb, sl, rr, em in combos:
            if lb != prev_lb and prev_lb is not None: print()
            prev_lb = lb
            bz, sz, ema = zone_cache[(lb, args.trend_filter)]
            label = (f"{pair}_{tf_label}_{args.trend_filter}_{em}"
                     f"_lb{lb}_sl{sl}_rr{str(rr).replace('.','')}")
            trades = run_backtest(
                df, bz, sz, ema, args.trend_filter, rr,
                pip_size, sl, max_sl,
                tf_hours=tf_hours, entry_mode=em)
            s = calc_stats(trades, label, rr, pair=pair,
                           tf_hours=tf_hours, sl_buffer=sl,
                           max_sl=max_sl, entry_mode=em,
                           bos_lookback=lb)
            all_stats.append(s)
            save_trade_csv(trades, trading_dir, label)
            rdd = round(s["net_return_%"]/s["max_dd_%"],2) \
                  if s.get("max_dd_%",0)>0 else 999
            print(f"  {lb:<4}{sl:>4}{rr:>5}{em:<8}"
                  f"{s['trades']:>7}"
                  f"{s.get('win_rate_%',0):>6.1f}"
                  f"{s.get('profit_factor',0):>6.2f}"
                  f"{s.get('net_return_%',0):>6.1f}"
                  f"{s.get('max_dd_%',0):>7.1f}{rdd:>7.2f}")

        print(hdr)
        df_m = pd.DataFrame([{k:v for k,v in s.items()
                               if not k.startswith("_")}
                              for s in all_stats if s["trades"] > 0])
        if len(df_m):
            best_ret = df_m.loc[df_m["net_return_%"].idxmax()]
            best_pf  = df_m.loc[df_m["profit_factor"].idxmax()]
            rdd_col  = df_m["net_return_%"]/df_m["max_dd_%"].clip(lower=0.1)
            best_rdd = df_m.loc[rdd_col.idxmax()]
            print(f"\n  ── Best combos ──")
            print(f"  Best return : {best_ret['label']}  "
                  f"{best_ret['net_return_%']}%")
            print(f"  Best PF     : {best_pf['label']}  "
                  f"PF={best_pf['profit_factor']}")
            print(f"  Best Ret/DD : {best_rdd['label']}  "
                  f"{rdd_col.max():.2f}×")
            df_m.to_csv(
                trading_dir/f"OB_BoS_Matrix_{pair}_{tf_label}.csv",
                index=False)
        return

    # ── SINGLE RUN ─────────────────────────────────────────────────────────
    if not args.csv:
        print("ERROR: --csv required (or use --multi / --matrix)"); return
    csv_path = trading_dir / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pair, tf_hours, pip_size, sl_buf, max_sl = \
        resolve_params(args, args.csv)
    tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")

    print(f"\nLoading: {args.csv}  "
          f"[{pair} {tf_label}  pip={pip_size}  "
          f"SL={sl_buf}p  max={max_sl}p  entry={args.entry}]")
    df = load_data(csv_path)
    print(f"  Bars: {len(df)}  |  "
          f"{df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[-1].date()}")

    print(f"Detecting OB_BoS zones [{args.trend_filter}]  "
          f"bos_lb={args.bos_lookback}  zone_filter={args.zone_filter}...")
    bz, sz, ema = detect_ob_bos_zones(
        df, args.bos_lookback, args.trend_filter,
        zone_filter=args.zone_filter,
        tf_hours=tf_hours, pip_size=pip_size)
    print(f"  Bull zones: {len(bz)}  Bear zones: {len(sz)}")

    print("Running backtest...")
    trades = run_backtest(df, bz, sz, ema, args.trend_filter, args.rr,
                          pip_size, sl_buf, max_sl,
                          tf_hours=tf_hours, entry_mode=args.entry)
    print(f"  Trades: {len(trades)}")

    label = (f"{pair}_{tf_label}_{args.trend_filter}_{args.entry}"
             f"_lb{args.bos_lookback}"
             f"_sl{sl_buf}_rr{str(args.rr).replace('.','')}")
    s = calc_stats(trades, label, args.rr, pair=pair,
                   tf_hours=tf_hours, sl_buffer=sl_buf,
                   max_sl=max_sl, entry_mode=args.entry,
                   bos_lookback=args.bos_lookback)
    print_summary(s, tf_label)
    save_trade_csv(trades, trading_dir, label)
    print(f"  Trade log → OB_BoS_Results_{label}.csv")


if __name__ == "__main__":
    main()
