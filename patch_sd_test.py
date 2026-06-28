"""
Patches SD_Test_Simple.py to add same-bar SL/TP check immediately after
each trades.append() call. Without this fix, trades that are stopped out
on the same bar as entry are incorrectly recorded as open (and later WIN).

Inserts a same-bar resolution block after each of the 6 trade entry points:
  Lines 337, 368, 409, 450, 481, 517  (zone["trade_taken"] = True lines)
"""

import re

script_path = r"C:\Trading\SD_Test_Simple.py"
backup_path = r"C:\Trading\SD_Test_Simple_BACKUP.py"

# Read original
with open(script_path, "r", encoding="utf-8") as f:
    content = f.read()

# Write backup first
with open(backup_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Backup written to {backup_path}")

# The same-bar check block to insert.
# Uses %INDENT% as a placeholder for the correct indentation.
# For buy trades:
BUY_SAME_BAR = """%INDENT%# ── Same-bar SL/TP check (fixes entry-bar miss bug) ──
%INDENT%_t = trades[-1]
%INDENT%if _t["result"] is None and _t["_dir"] == "buy":
%INDENT%    if bar_low <= _t["_sl"]:
%INDENT%        _t["result"] = "LOSS"; _t["exit_price"] = round(_t["_sl"], 5)
%INDENT%        _t["exit_time"] = bar_time
%INDENT%        _t["pnl_pips"] = round((_t["_sl"] - _t["entry_price"]) / pip_size, 1)
%INDENT%    elif bar_high >= _t["_tp"]:
%INDENT%        _t["result"] = "WIN";  _t["exit_price"] = round(_t["_tp"], 5)
%INDENT%        _t["exit_time"] = bar_time
%INDENT%        _t["pnl_pips"] = round((_t["_tp"] - _t["entry_price"]) / pip_size, 1)
"""

SELL_SAME_BAR = """%INDENT%# ── Same-bar SL/TP check (fixes entry-bar miss bug) ──
%INDENT%_t = trades[-1]
%INDENT%if _t["result"] is None and _t["_dir"] == "sell":
%INDENT%    if bar_high >= _t["_sl"]:
%INDENT%        _t["result"] = "LOSS"; _t["exit_price"] = round(_t["_sl"], 5)
%INDENT%        _t["exit_time"] = bar_time
%INDENT%        _t["pnl_pips"] = round((_t["entry_price"] - _t["_sl"]) / pip_size, 1)
%INDENT%    elif bar_low <= _t["_tp"]:
%INDENT%        _t["result"] = "WIN";  _t["exit_price"] = round(_t["_tp"], 5)
%INDENT%        _t["exit_time"] = bar_time
%INDENT%        _t["pnl_pips"] = round((_t["entry_price"] - _t["_tp"]) / pip_size, 1)
"""

lines = content.split("\n")
new_lines = []
insertions = 0

# Target patterns: line ends with zone["trade_taken"] = True
# followed by blank line — we insert after the trade_taken line
# We track which direction (buy/sell) based on proximity to _dir marker

i = 0
while i < len(lines):
    new_lines.append(lines[i])

    # Detect trade_taken = True lines that follow a trades.append block
    stripped = lines[i].strip()
    if stripped in ('zone["trade_taken"] = True',
                    'zone["trade_taken"] = True; zone["pending"] = False'):

        # Determine indentation from current line
        indent = len(lines[i]) - len(lines[i].lstrip())
        ind_str = " " * indent

        # Determine buy or sell by looking back for _dir in recent lines
        direction = "buy"
        for back in range(max(0, i-20), i):
            if '"_dir": "sell"' in lines[back] or "'_dir': 'sell'" in lines[back]:
                direction = "sell"
                break
            if '"_dir": "buy"' in lines[back] or "'_dir': 'buy'" in lines[back]:
                direction = "buy"
                break

        template = BUY_SAME_BAR if direction == "buy" else SELL_SAME_BAR
        block = template.replace("%INDENT%", ind_str)
        new_lines.append(block.rstrip("\n"))
        insertions += 1
        print(f"  Inserted {direction} same-bar check after line {i+1}: {lines[i].rstrip()}")

    i += 1

new_content = "\n".join(new_lines)

with open(script_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"\nDone. {insertions} insertion(s) made.")
print(f"Original backed up to: {backup_path}")
print(f"Now re-run your backtest to see corrected results.")
