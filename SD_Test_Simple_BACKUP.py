"""
SD_Test_Simple.py
Supply & Demand Zone Backtesting Strategy
Fully parametric — supports any pair, timeframe, SL settings.

─────────────────────────────────────────────────────────────────
RECOMMENDED CONFIG (validated on GBPUSD 2H):
    python SD_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv

  trend_filter = none  |  entry = bottom  |  rr = 1.5
─────────────────────────────────────────────────────────────────

SINGLE RUN:
    python SD_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv
    python SD_Test_Simple.py --csv PEPPERSTONE_XAUUSD__120.csv --entry bottom
    python SD_Test_Simple.py --csv PEPPERSTONE_USDJPY__60.csv  --entry top

ENTRY MATRIX (top vs mid vs bottom, 2 filters = 6 combos):
    python SD_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv --entry_matrix

FULL COMBO MATRIX (6 entry/filter combos x RR x age):
    python SD_Test_Simple.py --csv PEPPERSTONE_GBPUSD__120.csv --matrix

MULTI (all pairs x all TFs — CSVs must be in C:\\Trading):
    python SD_Test_Simple.py --multi
    python SD_Test_Simple.py --multi --entry bottom

ENTRY MODES:
    top    : Buy at zone TOP    / Sell at zone BOTTOM  (earliest entry)
    mid    : Buy at zone MID    / Sell at zone MID     (validated baseline)
    bottom : Buy at zone BOTTOM / Sell at zone TOP     (deepest, most confirmation)

TREND FILTERS:
    none         : All zones traded regardless of EMA direction (best overall)
    origination  : Zone only forms if explosive candle is with-trend

SL AUTO-SCALING BY TIMEFRAME (forex):
    1H → 7/20p   2H → 10/30p   4H → 15/45p   8H → 20/60p   1D → 30/90p

SL AUTO-SCALING FOR GOLD (XAUUSD):
    1H → 50/150p   2H → 80/250p   4H → 120/400p   8H → 200/600p   1D → 300/900p

PIP SIZE AUTO-DETECTION:
    JPY pairs → 0.01   |   XAUUSD → 0.10   |   All others → 0.0001

MULTI MODE PAIRS (original + new):
    Original : GBPUSD EURUSD AUDUSD NZDUSD USDCAD USDCHF
    New      : XAUUSD USDJPY AUDCAD AUDNZD

OUTPUTS:
    SD_Results_<pair>_<tf>_<label>.csv
    SD_Entry_Comparison_<pair>_<tf>.csv
    SD_Matrix_Results_<pair>_<tf>.csv
    SD_Multi_Summary.csv
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
ATR_MULT_SD     = 1.5
MAX_BASE_ATR    = 1.0
MAX_ZONES       = 30
BRISBANE_TZ     = pytz.timezone("Australia/Brisbane")
VALID_FILTERS   = ("origination", "entry", "both", "none")
VALID_ENTRIES   = ("top", "mid", "bottom")
RR_VALUES       = [1.5, 2.0, 2.5, 3.0]
AGE_VALUES      = [0, 30, 60, 120]
RR_WITH_VALUES  = [1.5, 1.75, 2.0, 2.5]
RR_CT_VALUES    = [1.5]

# ── PAIRS AND TFs ─────────────────────────────────────────────────────────────
MULTI_PAIRS_ORIGINAL = ["GBPUSD", "EURUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
MULTI_PAIRS_NEW      = ["XAUUSD", "USDJPY", "AUDCAD", "AUDNZD"]
MULTI_PAIRS          = MULTI_PAIRS_ORIGINAL + MULTI_PAIRS_NEW
MULTI_TF_MINS        = [60, 120, 240, 480, 1440]
MULTI_TF_LABELS      = {60:"1H", 120:"2H", 240:"4H", 480:"8H", 1440:"1D"}

# ── SL SCALING ────────────────────────────────────────────────────────────────
# Standard forex: (sl_buffer_pips, max_sl_pips)
SL_BY_TF = {
    1:  (7,   20),
    2:  (10,  30),
    4:  (15,  45),
    8:  (20,  60),
    24: (30,  90),
}
# Gold (XAUUSD) — much wider due to volatility
SL_BY_TF_GOLD = {
    1:  (50,  150),
    2:  (80,  250),
    4:  (120, 400),
    8:  (200, 600),
    24: (300, 900),
}
# ─────────────────────────────────────────────────────────────────────────────


def detect_pair_tf(csv_name: str):
    """Infer pair and TF hours from TradingView CSV filename."""
    name     = Path(csv_name).stem.upper()
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
    """Return pip size based on pair name."""
    if pair is None:
        return 0.0001
    p = pair.upper()
    if "JPY" in p:
        return 0.01
    if "XAU" in p or "GOLD" in p:
        return 0.10
    return 0.0001


def is_gold(pair: str) -> bool:
    if pair is None: return False
    return "XAU" in pair.upper() or "GOLD" in pair.upper()


def get_sl_defaults(pair: str, tf_hours: int) -> tuple:
    """Return (sl_buffer, max_sl) for pair/TF combo."""
    if is_gold(pair):
        return SL_BY_TF_GOLD.get(tf_hours, (80, 250))
    return SL_BY_TF.get(tf_hours, (10, 30))


def parse_args():
    p = argparse.ArgumentParser(description="S&D Zone Backtest")
    p.add_argument("--csv",             default=None,             help="CSV filename in C:\\Trading")
    p.add_argument("--pair",            default=None,             help="Pair label e.g. GBPUSD")
    p.add_argument("--tf_hours",        default=None, type=int,   help="Bar size in hours: 1,2,4,8,24")
    p.add_argument("--pip_size",        default=None, type=float, help="Pip size override")
    p.add_argument("--sl_buffer",       default=None, type=int,   help="Pips beyond zone edge for SL")
    p.add_argument("--max_sl",          default=None, type=int,   help="Max SL pips")
    p.add_argument("--trend_filter",    default="none",           choices=VALID_FILTERS)
    p.add_argument("--rr",              default=1.5,  type=float, help="RR for all trades (default 1.5)")
    p.add_argument("--rr_with",         default=None, type=float, help="RR override for with-trend trades")
    p.add_argument("--rr_counter",      default=None, type=float, help="RR override for counter-trend trades")
    p.add_argument("--min_age",         default=0,    type=int,   help="Min bars between zone formation and entry")
    p.add_argument("--entry",           default="bottom",         choices=VALID_ENTRIES,
                   help="Entry: top=zone top/bottom | mid=mid-zone | bottom=zone bottom/top (default: bottom)")
    p.add_argument("--entry_matrix",    action="store_true",
                   help="Run top vs mid vs bottom x none vs origination (6 combos)")
    p.add_argument("--matrix",          action="store_true",      help="Run RR x age matrix for single CSV")
    p.add_argument("--split_rr_matrix", action="store_true",      help="Run with-trend RR x counter RR matrix")
    p.add_argument("--multi",           action="store_true",      help="Run all pairs x all TFs in C:\\Trading")
    return p.parse_args()


def resolve_params(args, csv_name: str):
    """Resolve pair, tf_hours, pip_size, sl_buffer, max_sl from args + auto-detection."""
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
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def calc_atr(df, length):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def calc_ema(series, length):
    return series.ewm(span=length, min_periods=length, adjust=False).mean()


def detect_zones(df, trend_filter):
    atr       = calc_atr(df, ATR_LENGTH)
    trend_ema = calc_ema(df["close"], EMA_LENGTH)
    demand_zones, supply_zones = [], []
    check_at_origin = trend_filter in ("origination", "both")

    for i in range(2, len(df)):
        o, c    = df["open"][i], df["close"][i]
        atr_val = atr[i]
        if pd.isna(atr_val) or pd.isna(trend_ema[i]):
            continue
        body_size         = abs(c - o)
        is_bullish_origin = c > trend_ema[i]
        is_bearish_origin = c < trend_ema[i]
        is_explosive_up   = (c > o) and (body_size > atr_val * ATR_MULT_SD)
        is_explosive_dn   = (c < o) and (body_size > atr_val * ATR_MULT_SD)
        base_high         = max(df["high"][i - 1], df["high"][i - 2])
        base_low          = min(df["low"][i - 1],  df["low"][i - 2])
        is_tight          = (base_high - base_low) <= (atr_val * MAX_BASE_ATR)

        if is_explosive_up and is_tight:
            if (not check_at_origin) or is_bullish_origin:
                demand_zones.append({
                    "type": "demand", "top": base_high, "bottom": base_low,
                    "formed_bar": i, "formed_time": df["datetime"][i],
                    "origin_bullish": is_bullish_origin,
                    "is_active": True, "trade_taken": False,
                })
                if len(demand_zones) > MAX_ZONES:
                    demand_zones.pop(0)

        if is_explosive_dn and is_tight:
            if (not check_at_origin) or is_bearish_origin:
                supply_zones.append({
                    "type": "supply", "top": base_high, "bottom": base_low,
                    "formed_bar": i, "formed_time": df["datetime"][i],
                    "origin_bearish": is_bearish_origin,
                    "is_active": True, "trade_taken": False,
                })
                if len(supply_zones) > MAX_ZONES:
                    supply_zones.pop(0)

    return demand_zones, supply_zones, trend_ema


def run_backtest(df, demand_zones, supply_zones, trend_ema,
                 trend_filter, rr, pip_size, sl_buffer_pips, max_sl_pips,
                 rr_with=None, rr_counter=None, min_age=0, tf_hours=2,
                 entry_mode="bottom"):
    """
    Bar-by-bar backtest.

    entry_mode:
      'top'    — BUY at zone TOP / SELL at zone BOTTOM (earliest, shallowest)
                 Trigger+fill on same bar. Risk = top to SL (full zone + buffer)

      'mid'    — BUY at zone MID / SELL at zone MID (validated baseline)
                 Trigger: edge touch → pending limit at mid → fill when reached.
                 Risk = mid to SL

      'bottom' — BUY at zone BOTTOM / SELL at zone TOP (deepest, most confirmation)
                 Trigger+fill when price reaches zone bottom/top.
                 Risk = bottom to SL (buffer only — tightest)
    """
    d_zones        = copy.deepcopy(demand_zones)
    s_zones        = copy.deepcopy(supply_zones)
    trades         = []
    sl_buffer      = sl_buffer_pips * pip_size
    max_sl_dist    = max_sl_pips    * pip_size
    check_at_entry = trend_filter in ("entry", "both")
    rr_w           = rr_with    if rr_with    is not None else rr
    rr_ct          = rr_counter if rr_counter is not None else rr

    def get_rr(is_with): return rr_w if is_with else rr_ct

    for i in range(len(df)):
        bar_high  = df["high"][i]
        bar_low   = df["low"][i]
        bar_close = df["close"][i]
        bar_time  = df["datetime"][i]
        ema_now   = trend_ema[i]
        is_bullish_now = (bar_close > ema_now) if not pd.isna(ema_now) else True
        is_bearish_now = (bar_close < ema_now) if not pd.isna(ema_now) else True

        # ── DEMAND → BUY ──
        for zone in d_zones:
            if not zone["is_active"] or zone["formed_bar"] >= i:
                continue
            ztop, zbot = zone["top"], zone["bottom"]
            mid        = (ztop + zbot) / 2.0
            zone_age   = i - zone["formed_bar"]

            if bar_close < zbot:
                zone["is_active"] = False; zone["pending"] = False; continue

            if entry_mode == "top":
                # ── TOP MODE: immediate fill at zone TOP on first touch ──
                if not zone["trade_taken"] and bar_low <= ztop:
                    if zone_age < min_age: continue
                    if check_at_entry and not is_bullish_now:
                        zone["trade_taken"] = True; continue
                    entry = ztop
                    sl    = zbot - sl_buffer
                    rdist = entry - sl
                    if rdist > max_sl_dist:
                        zone["trade_taken"] = True; continue
                    iw = zone.get("origin_bullish", True)
                    tp = entry + rdist * get_rr(iw)
                    trades.append({
                        "entry_time":      bar_time,   "type":        "BUY",
                        "zone_type":       "Demand",   "entry_mode":  "top",
                        "trend_at_origin": "with" if iw else "counter",
                        "trend_at_entry":  "with" if is_bullish_now else "counter",
                        "zone_top":        round(ztop,  5), "zone_bottom": round(zbot, 5),
                        "entry_price":     round(entry, 5), "sl_price":    round(sl,   5),
                        "tp_price":        round(tp,    5), "rr_applied":  get_rr(iw),
                        "risk_pips":       round(rdist / pip_size, 1),
                        "zone_age_bars":   zone_age,
                        "zone_age_days":   round(zone_age * tf_hours / 24, 1),
                        "formed_time":     zone["formed_time"],
                        "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                        "_sl": sl, "_tp": tp, "_dir": "buy", "_ps": pip_size,
                    })
                    zone["trade_taken"] = True

            elif entry_mode == "bottom":
                # ── BOTTOM MODE: deeper entry at zone BOTTOM for buy ──
                if not zone["trade_taken"] and bar_low <= zbot:
                    if zone_age < min_age: continue
                    if check_at_entry and not is_bullish_now:
                        zone["trade_taken"] = True; continue
                    entry = zbot
                    sl    = zbot - sl_buffer
                    rdist = entry - sl
                    if rdist > max_sl_dist:
                        zone["trade_taken"] = True; continue
                    iw = zone.get("origin_bullish", True)
                    tp = entry + rdist * get_rr(iw)
                    trades.append({
                        "entry_time":      bar_time,   "type":        "BUY",
                        "zone_type":       "Demand",
                        "entry_mode":      "bottom",
                        "trend_at_origin": "with" if iw else "counter",
                        "trend_at_entry":  "with" if is_bullish_now else "counter",
                        "zone_top":        round(ztop, 5), "zone_bottom": round(zbot, 5),
                        "entry_price":     round(entry, 5), "sl_price":   round(sl, 5),
                        "tp_price":        round(tp, 5),    "rr_applied": get_rr(iw),
                        "risk_pips":       round(rdist / pip_size, 1),
                        "zone_age_bars":   zone_age,
                        "zone_age_days":   round(zone_age * tf_hours / 24, 1),
                        "formed_time":     zone["formed_time"],
                        "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                        "_sl": sl, "_tp": tp, "_dir": "buy", "_ps": pip_size,
                    })
                    zone["trade_taken"] = True

            else:
                # ── MID MODE: pending limit order at mid ──
                # Step 1 — edge touch → pending
                if not zone["trade_taken"] and not zone.get("pending", False):
                    if bar_low <= ztop:
                        if zone_age < min_age: continue
                        if check_at_entry and not is_bullish_now:
                            zone["trade_taken"] = True; continue
                        sl    = zbot - sl_buffer
                        rdist = mid - sl
                        if rdist > max_sl_dist:
                            zone["trade_taken"] = True; continue
                        zone.update({"pending": True, "pending_mid": mid,
                                     "pending_sl": sl, "pending_rdist": rdist,
                                     "pending_is_with": zone.get("origin_bullish", True),
                                     "pending_age": zone_age})

                # Step 2 — pending → fill when price trades through mid
                if not zone["trade_taken"] and zone.get("pending", False):
                    if bar_low <= zone["pending_mid"]:
                        mf, sl, rd = zone["pending_mid"], zone["pending_sl"], zone["pending_rdist"]
                        iw = zone["pending_is_with"]
                        tp = mf + rd * get_rr(iw)
                        trades.append({
                            "entry_time":      bar_time,   "type":        "BUY",
                            "zone_type":       "Demand",
                            "entry_mode":      "mid",
                            "trend_at_origin": "with" if iw else "counter",
                            "trend_at_entry":  "with" if is_bullish_now else "counter",
                            "zone_top":        round(ztop, 5), "zone_bottom": round(zbot, 5),
                            "entry_price":     round(mf, 5),   "sl_price":    round(sl, 5),
                            "tp_price":        round(tp, 5),   "rr_applied":  get_rr(iw),
                            "risk_pips":       round(rd / pip_size, 1),
                            "zone_age_bars":   zone["pending_age"],
                            "zone_age_days":   round(zone["pending_age"] * tf_hours / 24, 1),
                            "formed_time":     zone["formed_time"],
                            "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                            "_sl": sl, "_tp": tp, "_dir": "buy", "_ps": pip_size,
                        })
                        zone["trade_taken"] = True; zone["pending"] = False

        # ── SUPPLY → SELL ──
        for zone in s_zones:
            if not zone["is_active"] or zone["formed_bar"] >= i:
                continue
            ztop, zbot = zone["top"], zone["bottom"]
            mid        = (ztop + zbot) / 2.0
            zone_age   = i - zone["formed_bar"]

            if bar_close > ztop:
                zone["is_active"] = False; zone["pending"] = False; continue

            if entry_mode == "top":
                # ── TOP MODE: immediate fill at zone BOTTOM on first touch ──
                if not zone["trade_taken"] and bar_high >= zbot:
                    if zone_age < min_age: continue
                    if check_at_entry and not is_bearish_now:
                        zone["trade_taken"] = True; continue
                    entry = zbot
                    sl    = ztop + sl_buffer
                    rdist = sl - entry
                    if rdist > max_sl_dist:
                        zone["trade_taken"] = True; continue
                    iw = zone.get("origin_bearish", True)
                    tp = entry - rdist * get_rr(iw)
                    trades.append({
                        "entry_time":      bar_time,   "type":        "SELL",
                        "zone_type":       "Supply",   "entry_mode":  "top",
                        "trend_at_origin": "with" if iw else "counter",
                        "trend_at_entry":  "with" if is_bearish_now else "counter",
                        "zone_top":        round(ztop,  5), "zone_bottom": round(zbot, 5),
                        "entry_price":     round(entry, 5), "sl_price":    round(sl,   5),
                        "tp_price":        round(tp,    5), "rr_applied":  get_rr(iw),
                        "risk_pips":       round(rdist / pip_size, 1),
                        "zone_age_bars":   zone_age,
                        "zone_age_days":   round(zone_age * tf_hours / 24, 1),
                        "formed_time":     zone["formed_time"],
                        "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                        "_sl": sl, "_tp": tp, "_dir": "sell", "_ps": pip_size,
                    })
                    zone["trade_taken"] = True

            elif entry_mode == "bottom":
                # ── BOTTOM MODE: deeper entry at zone TOP for sell ──
                if not zone["trade_taken"] and bar_high >= ztop:
                    if zone_age < min_age: continue
                    if check_at_entry and not is_bearish_now:
                        zone["trade_taken"] = True; continue
                    entry = ztop
                    sl    = ztop + sl_buffer
                    rdist = sl - entry
                    if rdist > max_sl_dist:
                        zone["trade_taken"] = True; continue
                    iw = zone.get("origin_bearish", True)
                    tp = entry - rdist * get_rr(iw)
                    trades.append({
                        "entry_time":      bar_time,   "type":        "SELL",
                        "zone_type":       "Supply",
                        "entry_mode":      "bottom",
                        "trend_at_origin": "with" if iw else "counter",
                        "trend_at_entry":  "with" if is_bearish_now else "counter",
                        "zone_top":        round(ztop,  5), "zone_bottom": round(zbot, 5),
                        "entry_price":     round(entry, 5), "sl_price":    round(sl,   5),
                        "tp_price":        round(tp,    5), "rr_applied":  get_rr(iw),
                        "risk_pips":       round(rdist / pip_size, 1),
                        "zone_age_bars":   zone_age,
                        "zone_age_days":   round(zone_age * tf_hours / 24, 1),
                        "formed_time":     zone["formed_time"],
                        "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                        "_sl": sl, "_tp": tp, "_dir": "sell", "_ps": pip_size,
                    })
                    zone["trade_taken"] = True
                if not zone["trade_taken"] and not zone.get("pending", False):
                    if bar_high >= zbot:
                        if zone_age < min_age: continue
                        if check_at_entry and not is_bearish_now:
                            zone["trade_taken"] = True; continue
                        sl    = ztop + sl_buffer
                        rdist = sl - mid
                        if rdist > max_sl_dist:
                            zone["trade_taken"] = True; continue
                        zone.update({"pending": True, "pending_mid": mid,
                                     "pending_sl": sl, "pending_rdist": rdist,
                                     "pending_is_with": zone.get("origin_bearish", True),
                                     "pending_age": zone_age})

                if not zone["trade_taken"] and zone.get("pending", False):
                    if bar_high >= zone["pending_mid"]:
                        mf, sl, rd = zone["pending_mid"], zone["pending_sl"], zone["pending_rdist"]
                        iw = zone["pending_is_with"]
                        tp = mf - rd * get_rr(iw)
                        trades.append({
                            "entry_time":      bar_time,   "type":        "SELL",
                            "zone_type":       "Supply",
                            "entry_mode":      "mid",
                            "trend_at_origin": "with" if iw else "counter",
                            "trend_at_entry":  "with" if is_bearish_now else "counter",
                            "zone_top":        round(ztop, 5), "zone_bottom": round(zbot, 5),
                            "entry_price":     round(mf, 5),   "sl_price":    round(sl, 5),
                            "tp_price":        round(tp, 5),   "rr_applied":  get_rr(iw),
                            "risk_pips":       round(rd / pip_size, 1),
                            "zone_age_bars":   zone["pending_age"],
                            "zone_age_days":   round(zone["pending_age"] * tf_hours / 24, 1),
                            "formed_time":     zone["formed_time"],
                            "result": None, "exit_price": None, "exit_time": None, "pnl_pips": None,
                            "_sl": sl, "_tp": tp, "_dir": "sell", "_ps": pip_size,
                        })
                        zone["trade_taken"] = True; zone["pending"] = False

        # ── RESOLVE OPEN TRADES ──
        for t in trades:
            if t["result"] is not None: continue
            sl, tp, ps = t["_sl"], t["_tp"], t["_ps"]
            if t["_dir"] == "buy":
                if bar_low <= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,5); t["exit_time"]=bar_time
                    t["pnl_pips"]=round((sl - t["entry_price"]) / ps, 1)
                elif bar_high >= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,5); t["exit_time"]=bar_time
                    t["pnl_pips"]=round((tp - t["entry_price"]) / ps, 1)
            else:
                if bar_high >= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,5); t["exit_time"]=bar_time
                    t["pnl_pips"]=round((t["entry_price"] - sl) / ps, 1)
                elif bar_low <= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,5); t["exit_time"]=bar_time
                    t["pnl_pips"]=round((t["entry_price"] - tp) / ps, 1)

    return trades


def calc_stats(trades, label, rr_with, rr_counter, min_age=0,
               pair="", tf_hours=2, sl_buffer=10, max_sl=30, entry_mode="mid"):
    clean  = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trades]
    df_t   = pd.DataFrame(clean) if clean else pd.DataFrame()

    base = {"label": label, "pair": pair, "tf_hours": tf_hours,
            "entry_mode": entry_mode,
            "sl_buffer_pips": sl_buffer, "max_sl_pips": max_sl,
            "rr_with": rr_with, "rr_counter": rr_counter,
            "min_age_bars": min_age,
            "min_age_days": round(min_age * tf_hours / 24, 1),
            "trades": 0}

    if df_t.empty:
        return base

    closed  = df_t[df_t["result"].isin(["WIN", "LOSS"])]
    wins    = closed[closed["result"] == "WIN"]
    losses  = closed[closed["result"] == "LOSS"]
    open_t  = df_t[df_t["result"].isna()]
    n       = len(closed)
    if n == 0:
        return base

    wr      = len(wins) / n * 100
    gross_w = wins["pnl_pips"].sum()        if len(wins)   else 0
    gross_l = abs(losses["pnl_pips"].sum()) if len(losses) else 0
    pf      = gross_w / gross_l             if gross_l > 0 else 999.0
    total_p = gross_w - gross_l
    exp     = total_p / n

    wt   = closed[closed["trend_at_origin"] == "with"]
    ct   = closed[closed["trend_at_origin"] == "counter"]
    wt_w = wt[wt["result"]=="WIN"]; wt_l = wt[wt["result"]=="LOSS"]
    ct_w = ct[ct["result"]=="WIN"]; ct_l = ct[ct["result"]=="LOSS"]
    wt_wr   = len(wt_w)/len(wt)*100 if len(wt) else 0
    ct_wr   = len(ct_w)/len(ct)*100 if len(ct) else 0
    wt_pips = wt_w["pnl_pips"].sum() - abs(wt_l["pnl_pips"].sum()) if len(wt) else 0
    ct_pips = ct_w["pnl_pips"].sum() - abs(ct_l["pnl_pips"].sum()) if len(ct) else 0
    wt_pf   = wt_w["pnl_pips"].sum()/abs(wt_l["pnl_pips"].sum()) if len(wt_l)>0 else 999
    ct_pf   = ct_w["pnl_pips"].sum()/abs(ct_l["pnl_pips"].sum()) if len(ct_l)>0 else 999
    avg_age      = closed["zone_age_bars"].mean() if "zone_age_bars" in closed.columns else 0
    avg_age_days = closed["zone_age_days"].mean() if "zone_age_days" in closed.columns else 0

    account, equity, peak = 10000.0, 10000.0, 10000.0
    max_dd_d = max_dd_p = 0.0
    for _, row in closed.iterrows():
        risk_d  = equity * 0.01
        equity += row["pnl_pips"] * (risk_d / row["risk_pips"])
        if equity > peak: peak = equity
        dd = peak - equity; ddp = dd / peak * 100
        if dd > max_dd_d: max_dd_d = dd; max_dd_p = ddp

    return {**base,
        "trades":            n,
        "wins":              len(wins),
        "losses":            len(losses),
        "open":              len(open_t),
        "win_rate_%":        round(wr,      1),
        "profit_factor":     round(pf,      2),
        "total_pips":        round(total_p, 1),
        "exp_pips/trade":    round(exp,     1),
        "avg_zone_age_bars": round(avg_age,      1),
        "avg_zone_age_days": round(avg_age_days, 1),
        "wt_trades":         len(wt),  "wt_wins": len(wt_w),
        "wt_wr_%":           round(wt_wr,   1),
        "wt_pips":           round(wt_pips, 1), "wt_pf": round(wt_pf, 2),
        "ct_trades":         len(ct),  "ct_wins": len(ct_w),
        "ct_wr_%":           round(ct_wr,   1),
        "ct_pips":           round(ct_pips, 1), "ct_pf": round(ct_pf, 2),
        "final_equity_$":    round(equity,   2),
        "net_return_%":      round((equity - account)/account*100, 1),
        "max_dd_$":          round(max_dd_d, 2),
        "max_dd_%":          round(max_dd_p, 1),
        "_trades_list":      trades,
    }


def save_trade_csv(trades, output_dir, label):
    clean = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trades]
    if clean:
        pd.DataFrame(clean).to_csv(output_dir / f"SD_Results_{label}.csv", index=False)


def print_summary(s):
    entry = s.get("entry_mode", "mid").upper()
    entry_desc = {
        "TOP":    "Buy @ zone TOP    / Sell @ zone BOTTOM  (earliest)",
        "MID":    "Buy @ zone MID    / Sell @ zone MID     (baseline)",
        "BOTTOM": "Buy @ zone BOTTOM / Sell @ zone TOP     (deepest)",
    }.get(entry, entry)
    print(f"\n{'='*62}")
    print(f"  ▶ ENTRY MODE : {entry}  —  {entry_desc}")
    print(f"{'='*62}")
    print(f"  Pair: {s['pair']}  |  TF: {s['tf_hours']}H  |  "
          f"SL: {s['sl_buffer_pips']}pip buf / {s['max_sl_pips']}pip max")
    print(f"  filter=none  rr={s['rr_with']}  min_age={s['min_age_bars']} bars")
    print(f"{'='*62}")
    print(f"  Trades         : {s['trades']}  ({s['wins']}W / {s['losses']}L)")
    print(f"  Win rate       : {s['win_rate_%']}%")
    print(f"  Profit factor  : {s['profit_factor']}")
    print(f"  Total pips     : {s['total_pips']}")
    print(f"  Net return     : {s['net_return_%']}%  (${s['final_equity_$']:,.2f})")
    print(f"  Max DD         : {s['max_dd_%']}%  (${s['max_dd_$']:,.2f})")
    print(f"  Avg zone age   : {s['avg_zone_age_bars']} bars  ({s['avg_zone_age_days']} days)")
    print(f"  ── With-trend  : {s['wt_trades']}t  WR {s['wt_wr_%']}%  PF {s['wt_pf']}  {s['wt_pips']} pips")
    print(f"  ── Counter     : {s['ct_trades']}t  WR {s['ct_wr_%']}%  PF {s['ct_pf']}  {s['ct_pips']} pips")
    print(f"{'='*62}")


def run_single(csv_path, args, trading_dir):
    """Run a single CSV. Returns stats dict."""
    pair, tf_hours, pip_size, sl_buffer, max_sl = resolve_params(args, csv_path.name)
    tf_label   = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")
    entry_mode = args.entry

    print(f"\nLoading: {csv_path.name}  [{pair} {tf_label}  "
          f"pip={pip_size}  SL buf={sl_buffer}p  max={max_sl}p  entry={entry_mode}]")
    df = load_data(csv_path)
    print(f"  Bars: {len(df)}  |  {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")

    rr_with    = args.rr_with    if args.rr_with    is not None else args.rr
    rr_counter = args.rr_counter if args.rr_counter is not None else args.rr
    label      = f"{pair}_{tf_label}_{args.trend_filter}_{entry_mode}_RR{str(args.rr).replace('.','')}_age{args.min_age}"

    print(f"Detecting zones [{args.trend_filter}]...")
    dz, sz, ema = detect_zones(df, args.trend_filter)
    print(f"  Demand: {len(dz)}  Supply: {len(sz)}")

    print("Running backtest...")
    trades = run_backtest(df, dz, sz, ema, args.trend_filter, args.rr,
                          pip_size, sl_buffer, max_sl,
                          rr_with=rr_with, rr_counter=rr_counter,
                          min_age=args.min_age, tf_hours=tf_hours,
                          entry_mode=entry_mode)
    print(f"  Trades: {len(trades)}")

    s = calc_stats(trades, label, rr_with, rr_counter, min_age=args.min_age,
                   pair=pair, tf_hours=tf_hours, sl_buffer=sl_buffer,
                   max_sl=max_sl, entry_mode=entry_mode)
    print_summary(s)
    save_trade_csv(trades, trading_dir, label)
    print(f"  Trade log → SD_Results_{label}.csv")
    return s


def main():
    args        = parse_args()
    trading_dir = Path(r"C:\Trading")

    # ── MULTI MODE ────────────────────────────────────────────────────────────
    if args.multi:
        print(f"\nMULTI MODE — {len(MULTI_PAIRS)} pairs × {len(MULTI_TF_MINS)} TFs")
        print(f"  Pairs : {MULTI_PAIRS}")
        print(f"  TFs   : {[MULTI_TF_LABELS[t] for t in MULTI_TF_MINS]}")
        print(f"  RR={args.rr}  filter=none  min_age={args.min_age}\n")

        hdr = "─" * 100
        print(hdr)
        print(f"  {'Pair':<8} {'TF':<4} {'Trades':>7} {'WR%':>6} {'PF':>6} "
              f"{'Pips':>8} {'Ret%':>6} {'MaxDD%':>7} {'Ret/DD':>7} "
              f"{'SLbuf':>6} {'MaxSL':>6} {'CT':>4}")
        print(hdr)

        all_stats  = []
        not_found  = []

        for pair in MULTI_PAIRS:
            for tf_mins in MULTI_TF_MINS:
                tf_label = MULTI_TF_LABELS[tf_mins]
                # Try both D and 1440 suffixes for daily
                suffixes = ["D", str(tf_mins)] if tf_mins == 1440 else [str(tf_mins)]
                csv_path = None
                for sfx in suffixes:
                    candidate = trading_dir / f"PEPPERSTONE_{pair}__{sfx}.csv"
                    if candidate.exists():
                        csv_path = candidate
                        break

                if csv_path is None:
                    not_found.append(f"PEPPERSTONE_{pair}__{suffixes[0]}.csv")
                    continue

                tf_hours  = tf_mins // 60
                default_sl_buf, default_max_sl = get_sl_defaults(pair, tf_hours)
                sl_buffer = args.sl_buffer if args.sl_buffer is not None else default_sl_buf
                max_sl    = args.max_sl    if args.max_sl    is not None else default_max_sl
                pip_size  = args.pip_size  if args.pip_size  else get_pip_size(pair)

                try:
                    df = load_data(csv_path)
                    dz, sz, ema = detect_zones(df, "none")
                    trades = run_backtest(df, dz, sz, ema, "none", args.rr,
                                          pip_size, sl_buffer, max_sl,
                                          min_age=args.min_age, tf_hours=tf_hours,
                                          entry_mode=args.entry)
                    label = f"{pair}_{tf_label}_none_{args.entry}_RR{str(args.rr).replace('.','')}_age{args.min_age}"
                    s = calc_stats(trades, label, args.rr, args.rr,
                                   min_age=args.min_age, pair=pair,
                                   tf_hours=tf_hours, sl_buffer=sl_buffer,
                                   max_sl=max_sl, entry_mode=args.entry)
                    all_stats.append(s)
                    save_trade_csv(trades, trading_dir, label)
                    rdd = round(s["net_return_%"] / s["max_dd_%"], 2) if s.get("max_dd_%", 0) > 0 else 999
                    print(
                        f"  {pair:<8} {tf_label:<4} {s['trades']:>7} "
                        f"{s['win_rate_%']:>6.1f} {s['profit_factor']:>6.2f} "
                        f"{s['total_pips']:>8.1f} {s['net_return_%']:>6.1f} "
                        f"{s['max_dd_%']:>7.1f} {rdd:>7.2f} "
                        f"{sl_buffer:>6} {max_sl:>6} {s['ct_trades']:>4}"
                    )
                except Exception as e:
                    print(f"  {pair:<8} {tf_label:<4}  ERROR: {e}")

        print(hdr)
        if not_found:
            print(f"\n  Missing CSVs (skipped):")
            for f in not_found:
                print(f"    {f}")

        if all_stats:
            df_multi = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")}
                                      for s in all_stats])
            out = trading_dir / "SD_Multi_Summary.csv"
            df_multi.to_csv(out, index=False)
            print(f"\n  Multi summary → {out}")

            # Best per metric
            if len(df_multi) > 0:
                best_ret = df_multi.loc[df_multi["net_return_%"].idxmax()]
                best_pf  = df_multi.loc[df_multi["profit_factor"].idxmax()]
                best_dd  = df_multi.loc[df_multi["max_dd_%"].idxmin()]
                rdd_col  = df_multi["net_return_%"] / df_multi["max_dd_%"].clip(lower=0.1)
                best_rdd = df_multi.loc[rdd_col.idxmax()]
                print(f"\n  ── Best across all combos ──")
                print(f"  Best return   : {best_ret['pair']} {best_ret['tf_hours']}H  {best_ret['net_return_%']}%")
                print(f"  Best PF       : {best_pf['pair']} {best_pf['tf_hours']}H  PF={best_pf['profit_factor']}")
                print(f"  Lowest max DD : {best_dd['pair']} {best_dd['tf_hours']}H  {best_dd['max_dd_%']}%")
                print(f"  Best Ret/DD   : {best_rdd['pair']} {best_rdd['tf_hours']}H  {rdd_col.max():.2f}×")
        return

    # ── ENTRY MATRIX: top vs mid vs bottom × none vs origination (6 combos) ───
    if args.entry_matrix:
        if not args.csv:
            print("ERROR: --csv required for --entry_matrix mode"); return
        csv_path = trading_dir / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        pair, tf_hours, pip_size, sl_buffer, max_sl = resolve_params(args, args.csv)
        tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")
        combos   = [(em, tf) for em in VALID_ENTRIES for tf in ("none", "origination")]

        print(f"\nENTRY MATRIX  [{pair} {tf_label}]  —  6 combos  (RR={args.rr})")
        print(f"  SL buf={sl_buffer}p  max_sl={max_sl}p  pip={pip_size}  min_age={args.min_age}")
        print(f"  3 entries × 2 filters = {len(combos)} combinations\n")

        df = load_data(csv_path)
        print(f"  Bars: {len(df)}  |  {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")

        # Pre-detect zones per filter (reuse across entry modes)
        zone_cache = {}
        for tf_filt in ("none", "origination"):
            dz, sz, ema = detect_zones(df, tf_filt)
            zone_cache[tf_filt] = (dz, sz, ema)
            print(f"  Zones [{tf_filt}]: Demand={len(dz)}  Supply={len(sz)}")

        hdr = "─" * 100
        print(f"\n{hdr}")
        print(f"  {'Entry':<8} {'Filter':<13} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Pips':>8} "
              f"{'Ret%':>6} {'MaxDD%':>7} {'Ret/DD':>7} {'WT WR%':>8} {'CT WR%':>8}")
        print(hdr)

        all_stats = []
        prev_em   = None
        for em, tf_filt in combos:
            if em != prev_em and prev_em is not None:
                print()
            prev_em = em
            dz, sz, ema = zone_cache[tf_filt]
            label  = f"{pair}_{tf_label}_{tf_filt}_{em}_RR{str(args.rr).replace('.','')}_age{args.min_age}"
            trades = run_backtest(df, dz, sz, ema, tf_filt, args.rr,
                                  pip_size, sl_buffer, max_sl,
                                  min_age=args.min_age, tf_hours=tf_hours,
                                  entry_mode=em)
            s = calc_stats(trades, label, args.rr, args.rr, min_age=args.min_age,
                           pair=pair, tf_hours=tf_hours, sl_buffer=sl_buffer,
                           max_sl=max_sl, entry_mode=em)
            all_stats.append(s)
            rdd = round(s["net_return_%"] / s["max_dd_%"], 2) if s.get("max_dd_%", 0) > 0 else 999
            print(
                f"  {em:<8} {tf_filt:<13} {s['trades']:>7} {s['win_rate_%']:>6.1f} "
                f"{s['profit_factor']:>6.2f} {s['total_pips']:>8.1f} "
                f"{s['net_return_%']:>6.1f} {s['max_dd_%']:>7.1f} {rdd:>7.2f} "
                f"{s['wt_wr_%']:>8.1f} {s['ct_wr_%']:>8.1f}"
            )
            save_trade_csv(trades, trading_dir, label)

        print(hdr)
        # Best combo
        df_m     = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")} for s in all_stats])
        best_ret = df_m.loc[df_m["net_return_%"].idxmax()]
        best_pf  = df_m.loc[df_m["profit_factor"].idxmax()]
        best_rdd = df_m.loc[(df_m["net_return_%"]/df_m["max_dd_%"].clip(lower=0.1)).idxmax()]
        print(f"\n  ── Best combos ──")
        print(f"  Best return  : {best_ret['label']}  {best_ret['net_return_%']}%")
        print(f"  Best PF      : {best_pf['label']}   PF={best_pf['profit_factor']}")
        print(f"  Best Ret/DD  : {best_rdd['label']}")

        out = trading_dir / f"SD_Entry_Comparison_{pair}_{tf_label}.csv"
        df_m.to_csv(out, index=False)
        print(f"\n  Comparison CSV → {out}")
        return

    # ── MATRIX MODE ───────────────────────────────────────────────────────────
    if args.matrix:
        if not args.csv:
            print("ERROR: --csv required for --matrix mode"); return
        csv_path = trading_dir / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        pair, tf_hours, pip_size, sl_buffer, max_sl = resolve_params(args, args.csv)
        tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")
        combos   = [(rr, age) for rr in RR_VALUES for age in AGE_VALUES]

        print(f"\nMATRIX MODE  [{pair} {tf_label}]  —  "
              f"{len(RR_VALUES)} RR × {len(AGE_VALUES)} age = {len(combos)} combos")
        print(f"  SL buffer={sl_buffer}p  max_sl={max_sl}p  pip_size={pip_size}")
        ages_days = [round(a*tf_hours/24,1) for a in AGE_VALUES]
        print(f"  Age values: {AGE_VALUES} bars = {ages_days} days\n")

        df = load_data(csv_path)
        print(f"  Bars: {len(df)}  |  {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
        dz, sz, ema = detect_zones(df, "none")
        print(f"  Demand: {len(dz)}  Supply: {len(sz)}\n")

        hdr = "─" * 105
        print(hdr)
        print(f"  {'RR':<5} {'Age':>7} {'Days':>5} {'Trades':>7} "
              f"{'WR%':>6} {'PF':>6} {'Pips':>8} {'Ret%':>6} {'MaxDD%':>7} "
              f"{'AvgAge':>7} {'CT':>4} {'CT_WR%':>7}")
        print(hdr)

        all_stats = []; prev_rr = None
        for rr, age in combos:
            if rr != prev_rr:
                if prev_rr is not None: print()
                prev_rr = rr
            label  = f"{pair}_{tf_label}_none_RR{str(rr).replace('.','')}_age{age}"
            trades = run_backtest(df, dz, sz, ema, "none", rr,
                                  pip_size, sl_buffer, max_sl, min_age=age,
                                  tf_hours=tf_hours, entry_mode=args.entry)
            s      = calc_stats(trades, label, rr, rr, min_age=age,
                                 pair=pair, tf_hours=tf_hours,
                                 sl_buffer=sl_buffer, max_sl=max_sl,
                                 entry_mode=args.entry)
            all_stats.append(s)
            print(
                f"  {rr:<5} {age:>7} {s['min_age_days']:>5} {s['trades']:>7} "
                f"{s['win_rate_%']:>6.1f} {s['profit_factor']:>6.2f} "
                f"{s['total_pips']:>8.1f} {s['net_return_%']:>6.1f} "
                f"{s['max_dd_%']:>7.1f} {s['avg_zone_age_bars']:>7.1f} "
                f"{s['ct_trades']:>4} {s['ct_wr_%']:>7.1f}"
            )
            save_trade_csv(trades, trading_dir, label)

        print(f"\n{hdr}")
        df_m     = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")} for s in all_stats])
        best_ret = df_m.loc[df_m["net_return_%"].idxmax()]
        best_pf  = df_m.loc[df_m["profit_factor"].idxmax()]
        best_rdd = df_m.loc[(df_m["net_return_%"]/df_m["max_dd_%"].clip(lower=0.1)).idxmax()]
        print(f"\n  ── Best combos ──")
        print(f"  Best return  : {best_ret['label']}  {best_ret['net_return_%']}%")
        print(f"  Best PF      : {best_pf['label']}   PF={best_pf['profit_factor']}")
        print(f"  Best Ret/DD  : {best_rdd['label']}  "
              f"{best_rdd['net_return_%']}% / {best_rdd['max_dd_%']}% DD")
        out = trading_dir / f"SD_Matrix_Results_{pair}_{tf_label}.csv"
        df_m.to_csv(out, index=False)
        print(f"\n  Matrix CSV → {out}")
        return

    # ── SPLIT RR MATRIX ───────────────────────────────────────────────────────
    if args.split_rr_matrix:
        if not args.csv:
            print("ERROR: --csv required for --split_rr_matrix mode"); return
        csv_path = trading_dir / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        pair, tf_hours, pip_size, sl_buffer, max_sl = resolve_params(args, args.csv)
        tf_label = MULTI_TF_LABELS.get(tf_hours, f"{tf_hours}H")
        combos   = [(rw, rc) for rw in RR_WITH_VALUES for rc in RR_CT_VALUES]

        print(f"\nSPLIT RR MATRIX  [{pair} {tf_label}  age={args.min_age}]")
        df = load_data(csv_path)
        dz, sz, ema = detect_zones(df, "none")
        print(f"  Demand: {len(dz)}  Supply: {len(sz)}\n")

        hdr = "─" * 105
        print(hdr)
        print(f"  {'RRw':<5} {'RRct':<5} {'Trades':>7} {'WR%':>6} {'PF':>6} "
              f"{'Pips':>8} {'Ret%':>6} {'MaxDD%':>7}  WT  CT")
        print(hdr)

        all_stats = []
        for rw, rc in combos:
            label  = f"{pair}_{tf_label}_none_RRw{str(rw).replace('.','')}_RRct{str(rc).replace('.','')}_age{args.min_age}"
            trades = run_backtest(df, dz, sz, ema, "none", rw, pip_size, sl_buffer, max_sl,
                                  rr_with=rw, rr_counter=rc,
                                  min_age=args.min_age, tf_hours=tf_hours,
                                  entry_mode=args.entry)
            s      = calc_stats(trades, label, rw, rc, min_age=args.min_age,
                                 pair=pair, tf_hours=tf_hours,
                                 sl_buffer=sl_buffer, max_sl=max_sl,
                                 entry_mode=args.entry)
            all_stats.append(s)
            print(
                f"  {rw:<5} {rc:<5} {s['trades']:>7} {s['win_rate_%']:>6.1f} "
                f"{s['profit_factor']:>6.2f} {s['total_pips']:>8.1f} "
                f"{s['net_return_%']:>6.1f} {s['max_dd_%']:>7.1f}  "
                f"WT:{s['wt_trades']}t/{s['wt_wr_%']:.0f}%  "
                f"CT:{s['ct_trades']}t/{s['ct_wr_%']:.0f}%"
            )
            save_trade_csv(trades, trading_dir, label)

        print(hdr)
        df_m = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")} for s in all_stats])
        df_m.to_csv(trading_dir / f"SD_Matrix_Results_{pair}_{tf_label}_splitRR.csv", index=False)
        return

    # ── SINGLE RUN ────────────────────────────────────────────────────────────
    if not args.csv:
        print("ERROR: --csv required (or use --multi)"); return
    csv_path = trading_dir / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    run_single(csv_path, args, trading_dir)


if __name__ == "__main__":
    main()
