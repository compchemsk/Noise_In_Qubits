#!/usr/bin/env python3
"""
m1_01_extract_scans.py
Extracts every g_iso(Q) and A_iso(Q) scan from the original Colab notebook
(meeting22nd.py) WITHOUT re-typing any number, and writes a tidy CSV.

Output: data/module1/module1_scans.csv
columns: molecule, mode, quantity (g_iso | A_iso_MHz), Q, value, ref_value
"""
import re, ast, sys, os
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "module1", "scan_data_meeting22nd.py")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "module1", "module1_scans.csv")
lines = open(SRC, encoding="utf-8").read().split("\n")

def molecule_mode(header):
    h = header.replace("*", "").strip().lower()
    m = re.search(r"mode\s*(\d+)", h)
    mode = int(m.group(1)) if m else None
    if "ptempo" in h:  mol = "PTEMPO"
    elif "tempo" in h and "sam" not in h: mol = "TEMPO"
    else: mol = None
    return mol, mode

rows, mol, mode = [], None, None
in_sam = False
cur_kind, ref, x = None, None, None
i = 0
while i < len(lines):
    L = lines[i]
    s = L.strip()
    # section headers ------------------------------------------------------
    if s.startswith('"""') and ("mode" in s.lower() or "TEMPO SAM -11" in s):
        block = s
        j = i
        while block.count('"""') < 2 and j + 1 < len(lines):   # multi-line docstring
            j += 1; block += " " + lines[j].strip()
        if "TEMPO SAM -11" in block:
            in_sam = True
        if in_sam:
            m = re.search(r"Mode\s*(\d+)", block)
            mol, mode = "PTEMPO-alkyl", int(m.group(1)) if m else mode
        else:
            mol, mode = molecule_mode(block)
        i = j + 1; continue
    if s.startswith('"""') and in_sam and "mode" not in s.lower():
        in_sam = False; mol = None                                   # left SAM-11 block
    # reference values -------------------------------------------------------
    m = re.match(r"^(g0|A0)\s*=\s*([-0-9.eE+]+)", s)
    if m and mol:
        cur_kind = "g_iso" if m.group(1) == "g0" else "A_iso_MHz"
        ref = float(m.group(2))
    # arrays (may span several lines) ----------------------------------------
    m = re.match(r"^(x|y)\s*=\s*np\.array\(\[", s)
    if m and mol and cur_kind:
        buf = s; j = i
        while "])" not in buf:
            j += 1; buf += lines[j].strip()
        arr = ast.literal_eval(buf[buf.index("["): buf.index("]") + 1])
        if m.group(1) == "x":
            x = arr
        else:
            if len(arr) != len(x):
                raise ValueError(f"length mismatch {mol} mode {mode} {cur_kind}")
            for q, v in zip(x, arr):
                rows.append(dict(molecule=mol, mode=mode, quantity=cur_kind,
                                 Q=q, value=v, ref_value=ref))
            cur_kind = None
        i = j + 1; continue
    i += 1

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(df.groupby(["molecule", "quantity"])["mode"].unique())
print(f"{len(df)} points -> {OUT}")
