"""
LS_Test_Gold.py
Liquidity Sweep (LS) Strategy — Gold (XAUUSD) only, S&D zones
─────────────────────────────────────────────────────────────────
CONCEPT:
  Price breaks below a demand zone (or above a supply zone) —
  triggering retail stop losses (liquidity sweep) — then a strong
  reversal bar confirms institutional re-entry.

THREE-BAR SETUP:
  BUY (demand zone sweep):
    Bar 1: closes BELOW zone bottom  → zone broken / swept
    Bar 2: bullish (close > open) + volume > N × avg_volume(20)
    Bar 3: enter BUY at open

  SELL (supply zone sweep):
    Bar 1: closes ABOVE zone top     → zone broken / swept
    Bar 2: bearish (close < open) + volume > N × avg_volume(20)
    Bar 3: enter SELL at open

  SL: below bar 2 low (buy) / above bar 2 high (sell) + buffer
  TP: entry ± SL_distance × RR

TREND FILTER:
  With-trend only — bar 2 must be above EMA (buy) / below EMA (sell)

PARAMETERS ITERATED:
  Volume multiplier : 1.0, 1.2, 1.5  (× 20-bar avg volume)
  SL buffer pips    : 5, 10           (Gold pips = 0.10 each)
  RR                : 2.0, 2.5, 3.0, 3.5
  EMA length        : 50, 100, 200
  → 3 × 2 × 4 × 3 = 72 combinations per TF

USAGE:
  Single run (uses defaults):
    python LS_Test_Gold.py --csv PEPPERSTONE_XAUUSD__60.csv

  Full matrix (all 72 combos):
    python LS_Test_Gold.py --csv PEPPERSTONE_XAUUSD__60.csv --matrix

  All 3 TFs in one run:
    python LS_Test_Gold.py --multi

OUTPUTS:
  LS_Results_<tf>_<label>.csv        — trade log
  LS_Matrix_<tf>.csv                 — full matrix summary
  LS_Multi_Summary.csv               — cross-TF comparison
"""

import argparse
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import pytz

# ── CONSTANTS ────────────────────────────────────────────────────────────────
BRISBANE_TZ   = pytz.timezone("Australia/Brisbane")
ATR_LENGTH    = 14
ATR_MULT_SD   = 1.5
MAX_BASE_ATR  = 1.0
ATR_MULT_OB   = 1.5
SWING_LB      = 5       # OB BOS lookback bars
MAX_ZONES     = 100
VOL_AVG_BARS  = 20
PIP_SIZE      = 0.10        # Gold pip
PAIR          = "XAUUSD"

# ── PARAMETER GRID ────────────────────────────────────────────────────────────
VOL_MULTS   = [1.0, 1.2, 1.5]
SL_BUFFERS  = [5, 10]          # in pips
RR_VALUES   = [2.0, 2.5, 3.0, 3.5]
EMA_LENGTHS = [50, 100, 200]

# SL scaling by TF (Gold — wide)
SL_MAX_BY_TF = {1: 200, 2: 300, 4: 500}   # max SL in pips

MULTI_TFS = {
    60:  ("1H", "PEPPERSTONE_XAUUSD__60.csv"),
    120: ("2H", "PEPPERSTONE_XAUUSD__120.csv"),
    240: ("4H", "PEPPERSTONE_XAUUSD__240.csv"),
}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="LS Gold Backtest")
    p.add_argument("--csv",        default=None, help="CSV filename in C:\\Trading")
    p.add_argument("--vol_mult",   default=1.2,  type=float)
    p.add_argument("--sl_buffer",  default=5,    type=int)
    p.add_argument("--rr",         default=2.5,  type=float)
    p.add_argument("--ema_length", default=100,  type=int)
    p.add_argument("--matrix",     action="store_true",
                   help="Run full 72-combo matrix for single CSV")
    p.add_argument("--multi",      action="store_true",
                   help="Run matrix across all 3 TFs")
    return p.parse_args()


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
    # Volume column — use if available
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 1.0   # fallback — no volume filter
    return df


def calc_atr(df, length):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def calc_ema(series, length):
    return series.ewm(span=length, min_periods=length, adjust=False).mean()


def calc_vol_avg(series, bars=VOL_AVG_BARS):
    return series.rolling(window=bars, min_periods=bars).mean()


def detect_sd_zones(df):
    """
    Detect S&D demand and supply zones.
    Returns lists of zone dicts — same logic as SD_Test_Simple.py
    No trend filter here — trend filter applied at entry time.
    """
    atr = calc_atr(df, ATR_LENGTH)
    demand_zones = []
    supply_zones = []

    for i in range(2, len(df)):
        o, c   = df["open"][i], df["close"][i]
        atr_v  = atr[i]
        if pd.isna(atr_v): continue

        body  = abs(c - o)
        is_up = (c > o) and (body > atr_v * ATR_MULT_SD)
        is_dn = (c < o) and (body > atr_v * ATR_MULT_SD)

        base_high = max(df["high"][i-1], df["high"][i-2])
        base_low  = min(df["low"][i-1],  df["low"][i-2])
        is_tight  = (base_high - base_low) <= (atr_v * MAX_BASE_ATR)

        if is_up and is_tight:
            demand_zones.append({
                "top": base_high, "bottom": base_low,
                "formed_bar": i, "formed_time": df["datetime"][i],
                "is_active": True, "trade_taken": False,
            })
            if len(demand_zones) > MAX_ZONES:
                demand_zones.pop(0)

        if is_dn and is_tight:
            supply_zones.append({
                "top": base_high, "bottom": base_low,
                "formed_bar": i, "formed_time": df["datetime"][i],
                "is_active": True, "trade_taken": False,
            })
            if len(supply_zones) > MAX_ZONES:
                supply_zones.pop(0)

    return demand_zones, supply_zones


def detect_ob_zones(df):
    """
    Detect Order Block zones — same logic as OB_Test_Simple.py.
    Bullish OB: displacement up + BOS + bar[i-1] is DOWN candle
    Bearish OB: displacement dn + BOS + bar[i-1] is UP candle
    Zone = high/low of bar[i-1] (single candle)
    No trend filter — applied at entry time.
    """
    atr = calc_atr(df, ATR_LENGTH)
    demand_zones = []
    supply_zones = []

    for i in range(SWING_LB + 2, len(df)):
        o_i, c_i = df["open"][i], df["close"][i]
        atr_v    = atr[i]
        if pd.isna(atr_v): continue

        body      = abs(c_i - o_i)
        is_up     = (c_i > o_i) and (body > atr_v * ATR_MULT_OB)
        is_dn     = (c_i < o_i) and (body > atr_v * ATR_MULT_OB)

        if not is_up and not is_dn:
            continue

        # BOS levels
        swing_highs = [df["high"][i-k] for k in range(1, SWING_LB+1)]
        swing_lows  = [df["low"][i-k]  for k in range(1, SWING_LB+1)]
        highest_h   = max(swing_highs)
        lowest_l    = min(swing_lows)

        ob_high  = df["high"][i-1]
        ob_low   = df["low"][i-1]
        ob_is_dn = df["close"][i-1] < df["open"][i-1]  # red → bull OB
        ob_is_up = df["close"][i-1] > df["open"][i-1]  # green → bear OB

        # Bullish OB → demand zone
        if is_up and (c_i > highest_h) and ob_is_dn:
            demand_zones.append({
                "top": ob_high, "bottom": ob_low,
                "formed_bar": i, "formed_time": df["datetime"][i],
                "zone_source": "OB",
                "is_active": True, "trade_taken": False,
            })
            if len(demand_zones) > MAX_ZONES:
                demand_zones.pop(0)

        # Bearish OB → supply zone
        if is_dn and (c_i < lowest_l) and ob_is_up:
            supply_zones.append({
                "top": ob_high, "bottom": ob_low,
                "formed_bar": i, "formed_time": df["datetime"][i],
                "zone_source": "OB",
                "is_active": True, "trade_taken": False,
            })
            if len(supply_zones) > MAX_ZONES:
                supply_zones.pop(0)

    return demand_zones, supply_zones


def detect_all_zones(df, use_sd=True, use_ob=True):
    """
    Combine S&D and OB zones into merged demand/supply lists.
    Sort by formed_bar so backtest processes them chronologically.
    """
    d_all, s_all = [], []

    if use_sd:
        d_sd, s_sd = detect_sd_zones(df)
        for z in d_sd: z.setdefault("zone_source", "SD")
        for z in s_sd: z.setdefault("zone_source", "SD")
        d_all.extend(d_sd)
        s_all.extend(s_sd)

    if use_ob:
        d_ob, s_ob = detect_ob_zones(df)
        d_all.extend(d_ob)
        s_all.extend(s_ob)

    d_all.sort(key=lambda z: z["formed_bar"])
    s_all.sort(key=lambda z: z["formed_bar"])
    return d_all, s_all


def run_backtest(df, demand_zones, supply_zones,
                 vol_mult, sl_buffer_pips, rr, ema_length,
                 tf_hours=1, max_sl_pips=200):
    """
    Liquidity Sweep backtest.

    Scan bar by bar looking for 3-bar LS setup:
      BUY:
        bar[i-2] = bar 1 — close < demand zone bottom (sweep)
        bar[i-1] = bar 2 — bullish + strong volume
        bar[i]   = bar 3 — enter at open
        Trend:   bar[i-1] close > EMA

      SELL:
        bar[i-2] = bar 1 — close > supply zone top (sweep)
        bar[i-1] = bar 2 — bearish + strong volume
        bar[i]   = bar 3 — enter at open
        Trend:   bar[i-1] close < EMA
    """
    d_zones = copy.deepcopy(demand_zones)
    s_zones = copy.deepcopy(supply_zones)
    trades  = []

    ema     = calc_ema(df["close"], ema_length)
    vol_avg = calc_vol_avg(df["volume"])
    sl_buf  = sl_buffer_pips * PIP_SIZE
    max_sl  = max_sl_pips    * PIP_SIZE

    for i in range(3, len(df)):
        # bars
        b1_close  = df["close"][i-2]
        b1_low    = df["low"][i-2]
        b1_high   = df["high"][i-2]

        b2_open   = df["open"][i-1]
        b2_close  = df["close"][i-1]
        b2_high   = df["high"][i-1]
        b2_low    = df["low"][i-1]
        b2_vol    = df["volume"][i-1]
        b2_ema    = ema[i-1]
        b2_vavg   = vol_avg[i-1]

        b3_open   = df["open"][i]
        b3_time   = df["datetime"][i]

        if pd.isna(b2_ema) or pd.isna(b2_vavg):
            continue

        # Volume check
        vol_strong = (b2_vol >= vol_mult * b2_vavg) if b2_vavg > 0 else False

        # ── DEMAND ZONE BUY SWEEP ──────────────────────────────────────
        for zone in d_zones:
            if not zone["is_active"] or zone["trade_taken"]:
                continue
            if zone["formed_bar"] >= i - 2:
                continue

            zbot = zone["bottom"]
            ztop = zone["top"]

            # Invalidate if price closes above zone top
            if b1_close > ztop:
                zone["is_active"] = False
                continue

            # Bar 1: closed BELOW zone bottom (sweep)
            if b1_close >= zbot:
                continue

            # Bar 2: bullish + strong volume + above EMA (trend)
            b2_bull = b2_close > b2_open
            b2_trend = b2_close > b2_ema

            if not (b2_bull and vol_strong and b2_trend):
                continue

            # Entry at bar 3 open
            entry = b3_open
            sl    = b2_low - sl_buf
            rdist = entry - sl

            if rdist <= 0 or rdist > max_sl:
                zone["trade_taken"] = True
                continue

            tp = entry + rdist * rr

            trades.append({
                "entry_time":    b3_time,
                "type":          "BUY",
                "zone_type":     f"LS_Demand_{zone.get('zone_source','SD')}",
                "zone_top":      round(ztop,  2),
                "zone_bottom":   round(zbot,  2),
                "b1_close":      round(b1_close, 2),
                "b2_close":      round(b2_close, 2),
                "b2_volume":     round(b2_vol, 0),
                "b2_vol_avg":    round(b2_vavg, 0),
                "entry_price":   round(entry, 2),
                "sl_price":      round(sl,    2),
                "tp_price":      round(tp,    2),
                "rr_applied":    rr,
                "risk_pips":     round(rdist / PIP_SIZE, 1),
                "formed_time":   zone["formed_time"],
                "result":        None,
                "exit_price":    None,
                "exit_time":     None,
                "pnl_pips":      None,
                "_sl": sl, "_tp": tp, "_dir": "buy",
            })
            zone["trade_taken"] = True

        # ── SUPPLY ZONE SELL SWEEP ─────────────────────────────────────
        for zone in s_zones:
            if not zone["is_active"] or zone["trade_taken"]:
                continue
            if zone["formed_bar"] >= i - 2:
                continue

            zbot = zone["bottom"]
            ztop = zone["top"]

            # Invalidate if price closes below zone bottom
            if b1_close < zbot:
                zone["is_active"] = False
                continue

            # Bar 1: closed ABOVE zone top (sweep)
            if b1_close <= ztop:
                continue

            # Bar 2: bearish + strong volume + below EMA (trend)
            b2_bear  = b2_close < b2_open
            b2_trend = b2_close < b2_ema

            if not (b2_bear and vol_strong and b2_trend):
                continue

            # Entry at bar 3 open
            entry = b3_open
            sl    = b2_high + sl_buf
            rdist = sl - entry

            if rdist <= 0 or rdist > max_sl:
                zone["trade_taken"] = True
                continue

            tp = entry - rdist * rr

            trades.append({
                "entry_time":    b3_time,
                "type":          "SELL",
                "zone_type":     f"LS_Supply_{zone.get('zone_source','SD')}",
                "zone_top":      round(ztop,  2),
                "zone_bottom":   round(zbot,  2),
                "b1_close":      round(b1_close, 2),
                "b2_close":      round(b2_close, 2),
                "b2_volume":     round(b2_vol, 0),
                "b2_vol_avg":    round(b2_vavg, 0),
                "entry_price":   round(entry, 2),
                "sl_price":      round(sl,    2),
                "tp_price":      round(tp,    2),
                "rr_applied":    rr,
                "risk_pips":     round(rdist / PIP_SIZE, 1),
                "formed_time":   zone["formed_time"],
                "result":        None,
                "exit_price":    None,
                "exit_time":     None,
                "pnl_pips":      None,
                "_sl": sl, "_tp": tp, "_dir": "sell",
            })
            zone["trade_taken"] = True

        # ── RESOLVE OPEN TRADES ────────────────────────────────────────
        bar_h = df["high"][i]
        bar_l = df["low"][i]
        bar_t = df["datetime"][i]

        for t in trades:
            if t["result"] is not None: continue
            sl, tp = t["_sl"], t["_tp"]
            if t["_dir"] == "buy":
                if bar_l <= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,2)
                    t["exit_time"]=bar_t
                    t["pnl_pips"]=round((sl - t["entry_price"])/PIP_SIZE, 1)
                elif bar_h >= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,2)
                    t["exit_time"]=bar_t
                    t["pnl_pips"]=round((tp - t["entry_price"])/PIP_SIZE, 1)
            else:
                if bar_h >= sl:
                    t["result"]="LOSS"; t["exit_price"]=round(sl,2)
                    t["exit_time"]=bar_t
                    t["pnl_pips"]=round((t["entry_price"] - sl)/PIP_SIZE, 1)
                elif bar_l <= tp:
                    t["result"]="WIN";  t["exit_price"]=round(tp,2)
                    t["exit_time"]=bar_t
                    t["pnl_pips"]=round((t["entry_price"] - tp)/PIP_SIZE, 1)

    return trades


def calc_stats(trades, label, vol_mult, sl_buffer, rr, ema_length, tf_hours=1):
    clean  = [{k:v for k,v in t.items() if not k.startswith("_")} for t in trades]
    df_t   = pd.DataFrame(clean) if clean else pd.DataFrame()

    base = {
        "label": label, "tf_hours": tf_hours,
        "vol_mult": vol_mult, "sl_buffer_pips": sl_buffer,
        "rr": rr, "ema_length": ema_length, "trades": 0,
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
    net_p   = gross_w - gross_l
    exp     = net_p / n

    # Compounding equity
    account = 10000.0; equity = 10000.0; peak = 10000.0
    max_dd_d = max_dd_p = 0.0
    for _, row in closed.sort_values("entry_time").iterrows():
        risk_d  = equity * 0.01
        equity += row["pnl_pips"] * (risk_d / row["risk_pips"])
        if equity > peak: peak = equity
        dd = peak - equity; ddp = dd/peak*100
        if dd > max_dd_d: max_dd_d = dd; max_dd_p = ddp

    return {**base,
        "trades":          n,
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate_%":      round(wr,      1),
        "profit_factor":   round(pf,      2),
        "total_pips":      round(net_p,   1),
        "exp_pips/trade":  round(exp,     1),
        "net_return_%":    round((equity-account)/account*100, 1),
        "final_equity_$":  round(equity,  2),
        "max_dd_%":        round(max_dd_p,1),
        "max_dd_$":        round(max_dd_d,2),
        "_trades_list":    trades,
    }


def save_csv(trades, out_dir, label):
    clean = [{k:v for k,v in t.items() if not k.startswith("_")} for t in trades]
    if clean:
        pd.DataFrame(clean).to_csv(out_dir/f"LS_Results_{label}.csv", index=False)


def print_summary(s, tf_label=""):
    print(f"\n{'='*60}")
    print(f"  ▶ LIQUIDITY SWEEP — XAUUSD {tf_label}")
    print(f"{'='*60}")
    print(f"  vol_mult={s['vol_mult']}  sl_buf={s['sl_buffer_pips']}p  "
          f"rr={s['rr']}  ema={s['ema_length']}")
    print(f"{'='*60}")
    print(f"  Trades        : {s['trades']}  ({s.get('wins',0)}W / {s.get('losses',0)}L)")
    print(f"  Win rate      : {s.get('win_rate_%',0)}%")
    print(f"  Profit factor : {s.get('profit_factor',0)}")
    print(f"  Total pips    : {s.get('total_pips',0)}")
    print(f"  Net return    : {s.get('net_return_%',0)}%  "
          f"(${s.get('final_equity_$',10000):,.2f})")
    print(f"  Max DD        : {s.get('max_dd_%',0)}%")
    print(f"{'='*60}")


def run_matrix(df, tf_hours, tf_label, trading_dir, max_sl_pips):
    """Run all 72 combos and print summary table."""
    combos = [
        (vm, sl, rr, el)
        for vm in VOL_MULTS
        for sl in SL_BUFFERS
        for rr in RR_VALUES
        for el in EMA_LENGTHS
    ]

    print(f"\nLS MATRIX — XAUUSD {tf_label}  —  {len(combos)} combos")
    print(f"  Bars: {len(df)}  |  "
          f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}\n")

    hdr = "─"*95
    print(hdr)
    print(f"  {'VM':<5}{'SL':>4}{'RR':>5}{'EMA':>5}{'Tr':>5}{'WR%':>7}"
          f"{'PF':>6}{'Pips':>8}{'Ret%':>7}{'DD%':>6}{'Ret/DD':>8}")
    print(hdr)

    all_stats = []
    prev_vm   = None

    # Pre-detect zones once (SD + OB combined)
    dz, sz = detect_all_zones(df, use_sd=True, use_ob=True)
    print(f"  Zones — Demand: {len(dz)}  Supply: {len(sz)}  "
          f"(SD + OB combined)")

    for vm, sl, rr, el in combos:
        if vm != prev_vm and prev_vm is not None:
            print()
        prev_vm = vm

        label  = f"XAUUSD_{tf_label}_vm{str(vm).replace('.','')}_sl{sl}_rr{str(rr).replace('.','')}_ema{el}"
        trades = run_backtest(df, dz, sz, vm, sl, rr, el,
                              tf_hours=tf_hours, max_sl_pips=max_sl_pips)
        s = calc_stats(trades, label, vm, sl, rr, el, tf_hours)
        all_stats.append(s)

        rdd = round(s.get("net_return_%",0)/s["max_dd_%"],2) if s.get("max_dd_%",0)>0 else 999
        print(f"  {vm:<5}{sl:>4}{rr:>5}{el:>5}"
              f"{s['trades']:>5}{s.get('win_rate_%',0):>7.1f}"
              f"{s.get('profit_factor',0):>6.2f}{s.get('total_pips',0):>8.1f}"
              f"{s.get('net_return_%',0):>7.1f}{s.get('max_dd_%',0):>6.1f}{rdd:>8.2f}")

        save_csv(trades, trading_dir, label)

    print(hdr)

    df_m = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")}
                          for s in all_stats if s["trades"] > 0])

    if len(df_m):
        best_ret = df_m.loc[df_m["net_return_%"].idxmax()]
        best_pf  = df_m.loc[df_m["profit_factor"].idxmax()]
        rdd_col  = df_m["net_return_%"]/df_m["max_dd_%"].clip(lower=0.1)
        best_rdd = df_m.loc[rdd_col.idxmax()]
        print(f"\n  ── Best combos ──")
        print(f"  Best return : {best_ret['label']}  {best_ret['net_return_%']}%")
        print(f"  Best PF     : {best_pf['label']}   PF={best_pf['profit_factor']}")
        print(f"  Best Ret/DD : {best_rdd['label']}  {rdd_col.max():.2f}×")

        out = trading_dir / f"LS_Matrix_XAUUSD_{tf_label}.csv"
        df_m.to_csv(out, index=False)
        print(f"\n  Matrix CSV → {out}")

    return all_stats


def main():
    args        = parse_args()
    trading_dir = Path(r"C:\Trading")

    # ── MULTI: all 3 TFs ──────────────────────────────────────────────────
    if args.multi:
        all_multi = []
        for tf_mins, (tf_label, csv_name) in MULTI_TFS.items():
            csv_path = trading_dir / csv_name
            if not csv_path.exists():
                print(f"  Missing: {csv_name} — skipping")
                continue
            tf_hours   = tf_mins // 60
            max_sl_pip = SL_MAX_BY_TF.get(tf_hours, 200)
            print(f"\n{'═'*60}")
            print(f"  XAUUSD {tf_label}")
            print(f"{'═'*60}")
            df = load_data(csv_path)
            stats = run_matrix(df, tf_hours, tf_label, trading_dir, max_sl_pip)
            all_multi.extend(stats)

        # Cross-TF best combos
        df_all = pd.DataFrame([{k:v for k,v in s.items() if not k.startswith("_")}
                                 for s in all_multi if s["trades"] > 0])
        if len(df_all):
            out = trading_dir / "LS_Multi_Summary.csv"
            df_all.to_csv(out, index=False)
            rdd_col = df_all["net_return_%"]/df_all["max_dd_%"].clip(lower=0.1)
            best    = df_all.loc[rdd_col.idxmax()]
            print(f"\n  ── Best across all TFs ──")
            print(f"  {best['label']}  Ret={best['net_return_%']}%  "
                  f"PF={best['profit_factor']}  DD={best['max_dd_%']}%  "
                  f"Ret/DD={rdd_col.max():.2f}×")
            print(f"\n  Multi summary → {out}")
        return

    # ── SINGLE CSV ────────────────────────────────────────────────────────
    if not args.csv:
        print("ERROR: --csv required (or use --multi)"); return
    csv_path = trading_dir / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Detect TF from filename
    tf_hours = 1
    tf_label = "1H"
    for mins, (lbl, fname) in MULTI_TFS.items():
        if fname.lower() in args.csv.lower():
            tf_hours = mins // 60
            tf_label = lbl
            break

    max_sl_pip = SL_MAX_BY_TF.get(tf_hours, 200)

    print(f"\nLoading: {args.csv}  [XAUUSD {tf_label}]")
    df = load_data(csv_path)
    print(f"  Bars: {len(df)}  |  "
          f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")

    if args.matrix:
        run_matrix(df, tf_hours, tf_label, trading_dir, max_sl_pip)
        return

    # Single run with specified params
    print(f"Detecting S&D + OB zones...")
    dz, sz = detect_all_zones(df, use_sd=True, use_ob=True)
    print(f"  Demand: {len(dz)}  Supply: {len(sz)}  (SD + OB combined)")

    print(f"Running LS backtest  "
          f"[vol_mult={args.vol_mult}  sl={args.sl_buffer}p  "
          f"rr={args.rr}  ema={args.ema_length}]...")
    trades = run_backtest(df, dz, sz, args.vol_mult, args.sl_buffer,
                          args.rr, args.ema_length,
                          tf_hours=tf_hours, max_sl_pips=max_sl_pip)
    print(f"  Trades: {len(trades)}")

    label = (f"XAUUSD_{tf_label}_vm{str(args.vol_mult).replace('.','')}"
             f"_sl{args.sl_buffer}_rr{str(args.rr).replace('.','')}"
             f"_ema{args.ema_length}")
    s = calc_stats(trades, label, args.vol_mult, args.sl_buffer,
                   args.rr, args.ema_length, tf_hours)
    print_summary(s, tf_label)
    save_csv(trades, trading_dir, label)
    print(f"  Trade log → LS_Results_{label}.csv")


if __name__ == "__main__":
    main()
