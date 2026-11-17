#!/usr/bin/env python3
"""
reduce_module2_trajectories.py
--------------------------------
Shrinks the Module II MD data (parm7 + xtc) to a compact, analysis-ready
package without losing anything needed for the manuscript.

What it keeps
  * every atom of the organic monolayer (S, chains, triazole, TEMPO heads)
  * the top Au layer only (for head-group/S--Au geometry); bulk Au is dropped
  * the periodic box for every frame (needed for correct neighbour search)
  * frames at a chosen stride (default: every 10th saved frame)

What it writes (one folder per trajectory)
  * <name>_first_frame.pdb      : full-precision reference structure
  * <name>_topology.csv         : index, name, type, element, resid, resname, charge, mass
  * <name>_coords.npz           : float32 coordinates (n_frames, n_atoms, 3), box, time

Usage (on your own computer, where MDAnalysis is installed):
    pip install MDAnalysis numpy
    python reduce_module2_trajectories.py  --stride 10
Then zip the output folder "module2_reduced" (expected size: a few hundred MB).
"""
import argparse, os, glob
import numpy as np
import MDAnalysis as mda

p = argparse.ArgumentParser()
p.add_argument("--indir", default=".", help="folder containing *.parm7 and *.xtc")
p.add_argument("--outdir", default="module2_reduced")
p.add_argument("--stride", type=int, default=10, help="keep every n-th frame")
p.add_argument("--au_layer_tol", type=float, default=1.0,
               help="Au atoms within this distance (A) of the topmost Au z are kept")
args = p.parse_args()
os.makedirs(args.outdir, exist_ok=True)

pairs = []
for top in sorted(glob.glob(os.path.join(args.indir, "*.parm7"))):
    stem = os.path.splitext(os.path.basename(top))[0]
    trajs = sorted(glob.glob(os.path.join(args.indir, stem + "*.xtc")))
    for tr in trajs:
        pairs.append((top, tr))
if not pairs:
    raise SystemExit("No matching *.parm7 / *.xtc pairs found in " + args.indir)

for top, tr in pairs:
    name = os.path.splitext(os.path.basename(tr))[0]
    print(f"\n=== {name}")
    u = mda.Universe(top, tr)
    au = u.select_atoms("name AU* Au* or element Au or resname AU AUM GLD")
    org = u.atoms - au
    print(f"  atoms total {len(u.atoms)} | Au {len(au)} | organic {len(org)} | frames {len(u.trajectory)}")

    # top Au layer from first frame
    u.trajectory[0]
    if len(au):
        zmax = au.positions[:, 2].max()
        au_top = au[au.positions[:, 2] > zmax - args.au_layer_tol]
    else:
        au_top = au
    keep = org + au_top
    print(f"  keeping {len(keep)} atoms (top Au layer: {len(au_top)})")

    od = os.path.join(args.outdir, name); os.makedirs(od, exist_ok=True)
    keep.write(os.path.join(od, f"{name}_first_frame.pdb"))

    def attr(a, key, default=""):
        try:
            return getattr(a, key)
        except Exception:
            return default
    with open(os.path.join(od, f"{name}_topology.csv"), "w") as f:
        f.write("index,name,type,element,resid,resname,charge,mass\n")
        for a in keep:
            f.write(f"{a.index},{a.name},{attr(a,'type')},{attr(a,'element')},"
                    f"{a.resid},{a.resname},{attr(a,'charge',0.0):.5f},{a.mass:.4f}\n")

    frames = range(0, len(u.trajectory), args.stride)
    X = np.empty((len(frames), len(keep), 3), dtype=np.float32)
    B = np.empty((len(frames), 6), dtype=np.float32)
    T = np.empty(len(frames), dtype=np.float32)
    for k, i in enumerate(frames):
        ts = u.trajectory[i]
        X[k] = keep.positions
        B[k] = ts.dimensions if ts.dimensions is not None else np.zeros(6)
        T[k] = ts.time
    np.savez_compressed(os.path.join(od, f"{name}_coords.npz"),
                        coords=X, box=B, time_ps=T,
                        source_top=os.path.basename(top), source_traj=os.path.basename(tr),
                        stride=args.stride)
    print(f"  saved {len(frames)} frames -> {od}")
    if np.allclose(B, 0):
        print("  WARNING: no box information in this xtc; periodic neighbour search will not be possible.")

print("\nDone. Zip the folder:", args.outdir)
