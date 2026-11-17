#!/usr/bin/env bash
# m2_00b_decode_trajectories.sh
# Builds the stand-alone XTC decoder and converts the six production
# trajectories into flat float32 arrays read by m2_00_io.load_traj().
#
#   TSAM_MD_RAW : folder with {100K,300K}/rep{1,2,3}/TEMPO-SAM-11.xtc   (default ../md_raw)
#   TSAM_MD_BIN : output folder                                          (default ../md_bin)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="${TSAM_MD_RAW:-$HERE/../md_raw}"
BIN="${TSAM_MD_BIN:-$HERE/../md_bin}"
mkdir -p "$BIN"
gcc -O2 -o "$HERE/xtc_reader/xtc2bin" "$HERE/xtc_reader/xtc2bin.c"
for T in 300 100; do
  for r in 1 2 3; do
    echo "decoding ${T} K replica ${r}"
    "$HERE/xtc_reader/xtc2bin" "$RAW/${T}K/rep${r}/TEMPO-SAM-11.xtc" "$BIN/T${T}_r${r}"
  done
done
echo "done: $(ls "$BIN" | wc -l) files in $BIN"
