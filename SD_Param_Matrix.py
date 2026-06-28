"""
SD_Param_Matrix.py  v5
Fix: sets PYTHONIOENCODING=utf-8 in subprocess env to handle Unicode arrow character
"""

import subprocess, sys, os, csv, re, glob
from itertools import product

TRADING_DIR = r"C:\Trading"
SCRIPT      = os.path.join(TRADING_DIR, "SD_Test_Simple.py")
OUTPUT_CSV  = os.path.join(TRADING_DIR, "SD_Matrix_Summary.csv")
PYTHON      = sys.executable

SL_BUFFERS  = [10, 15, 20]
RR_VALUES   = [1.5, 1.8, 2.0]
FILTERS     = ["none", "origination"]
ENTRY       = "bottom"

csvs = sorted(glob.glob(os.path.join(TRADING_DIR, "PEPPERSTONE_*__*.csv")))
print(f"Found {len(csvs)} CSV files")

def parse_output(text):
    d = {}

    m = re.search(r"Loading:\s+(\S+\.csv)\s+\[(\w+)\s+(\w+)", text)
    if m:
        d["csv"]  = m.group(1)
        d["pair"] = m.group(2)
        d["tf"]   = m.group(3)

    m = re.search(r"Bars:\s+([\d,]+)", text)
    if m:
        d["bars"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Trades\s+:\s+(\d+)\s+\((\d+)W\s*/\s*(\d+)L\)", text)
    if m:
        d["trades"]  = int(m.group(1))
        d["wins"]    = int(m.group(2))
        d["losses"]  = int(m.group(3))

    m = re.search(r"Win rate\s+:\s+([\d.]+)%", text)
    if m: d["win_rate"] = float(m.group(1))

    m = re.search(r"Profit factor\s+:\s+([\d.]+)", text)
    if m: d["profit_factor"] = float(m.group(1))

    m = re.search(r"Total pips\s+:\s+([\d.]+)", text)
    if m: d["total_pips"] = float(m.group(1))

    m = re.search(r"Net return\s+:\s+([\d.]+)%", text)
    if m: d["net_return"] = float(m.group(1))

    m = re.search(r"Max DD\s+:\s+([\d.]+)%", text)
    if m: d["max_dd"] = float(m.group(1))

    m = re.search(r"Avg zone age\s+:\s+([\d.]+) bars", text)
    if m: d["avg_zone_age"] = float(m.group(1))

    m = re.search(r"With-trend\s+:\s+(\d+)t\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+([\d.]+)\s+pips", text)
    if m:
        d["wt_trades"] = int(m.group(1))
        d["wt_wr"]     = float(m.group(2))
        d["wt_pf"]     = float(m.group(3))
        d["wt_pips"]   = float(m.group(4))

    m = re.search(r"Counter\s+:\s+(\d+)t\s+WR\s+([\d.]+)%\s+PF\s+([\d.]+)\s+([\d.]+)\s+pips", text)
    if m:
        d["ct_trades"] = int(m.group(1))
        d["ct_wr"]     = float(m.group(2))
        d["ct_pf"]     = float(m.group(3))
        d["ct_pips"]   = float(m.group(4))

    return d

COLS = [
    "pair","tf","filter","sl_buffer","rr","entry",
    "trades","wins","losses","win_rate","profit_factor",
    "total_pips","net_return","max_dd","avg_zone_age",
    "wt_trades","wt_wr","wt_pf","wt_pips",
    "ct_trades","ct_wr","ct_pf","ct_pips",
    "bars","csv"
]

total_runs = len(csvs) * len(SL_BUFFERS) * len(RR_VALUES) * len(FILTERS)
print(f"Total runs: {total_runs}")
print(f"Output: {OUTPUT_CSV}\n")

# Force UTF-8 in subprocess environment — fixes Unicode arrow crash on Windows
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

results = []
run_num = 0

for csv_file, sl, rr, filt in product(csvs, SL_BUFFERS, RR_VALUES, FILTERS):
    run_num += 1
    csv_name = os.path.basename(csv_file)

    cmd = [
        PYTHON, SCRIPT,
        "--csv",          csv_name,
        "--entry",        ENTRY,
        "--trend_filter", filt,
        "--sl_buffer",    str(sl),
        "--rr",           str(rr),
    ]

    print(f"[{run_num}/{total_runs}] {csv_name:<35} SL={sl:>2}  RR={rr}  "
          f"filter={filt:<12}", end=" ... ", flush=True)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=TRADING_DIR,
            timeout=120,
            env=env,
            encoding="utf-8",
            errors="replace"
        )
        output = proc.stdout + proc.stderr
        row = parse_output(output)
        row["filter"]    = filt
        row["sl_buffer"] = sl
        row["rr"]        = rr
        row["entry"]     = ENTRY

        t  = row.get("trades",  "?")
        wr = row.get("win_rate","?")
        pf = row.get("profit_factor","?")
        print(f"t={t}  WR={wr}%  PF={pf}")
        results.append(row)

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. {len(results)} rows written to:\n   {OUTPUT_CSV}")

# Top 15 by PF
valid = [r for r in results if r.get("profit_factor") is not None and (r.get("trades") or 0) >= 10]
top = sorted(valid, key=lambda x: x["profit_factor"], reverse=True)[:15]
print(f"\nTop 15 by Profit Factor (min 10 trades):")
print(f"  {'pair':<8} {'tf':<4} {'SL':>4} {'RR':>5}  {'filter':<14} "
      f"{'t':>4}  {'WR':>6}  {'PF':>6}  {'pips':>8}  {'ret%':>6}  {'DD%':>5}")
print("  " + "-"*80)
for r in top:
    print(f"  {str(r.get('pair','?')):<8} {str(r.get('tf','?')):<4} "
          f"{str(r.get('sl_buffer','?')):>4}  {str(r.get('rr','?')):>4}  "
          f"{str(r.get('filter','?')):<14} "
          f"{str(r.get('trades','?')):>4}  "
          f"{str(r.get('win_rate','?')):>5}%  "
          f"{str(r.get('profit_factor','?')):>6}  "
          f"{str(r.get('total_pips','?')):>8}  "
          f"{str(r.get('net_return','?')):>6}  "
          f"{str(r.get('max_dd','?')):>5}")
