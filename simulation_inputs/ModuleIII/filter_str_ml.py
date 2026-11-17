#ls struct_*.xyz | sort -V 
import os
import shutil
import numpy as np

# ============================================================
# USER INPUT
# ============================================================

input_dir = "generated_structures"
output_dir = "filtered_structures"

# Number of atoms in ONE monomer
natoms_monomer = 29

# vdW scale factor
scale = 0.80

# Bondi van der Waals radii (Å)
vdw = {
    "H": 1.30,
    "C": 1.70,
    "N": 1.55,
    "O": 1.50
}

# Skip these intermolecular atom pairs (optional)
# Example:
# skip_pairs = {(28,28)}   # ignore O1-O2
skip_pairs = set()

os.makedirs(output_dir, exist_ok=True)

# ============================================================

accepted = 0
rejected = 0

log = open("rejected_structures.txt", "w")

xyz_files = sorted([f for f in os.listdir(input_dir)
                    if f.endswith(".xyz")])

for xyz in xyz_files:

    path = os.path.join(input_dir, xyz)

    with open(path) as f:
        lines = f.readlines()

    natoms = int(lines[0])

    elements = []
    coords = []

    for line in lines[2:]:

        sp = line.split()

        elements.append(sp[0])

        coords.append([float(sp[1]),
                       float(sp[2]),
                       float(sp[3])])

    coords = np.array(coords)

    mon1 = coords[:natoms_monomer]
    mon2 = coords[natoms_monomer:]

    elem1 = elements[:natoms_monomer]
    elem2 = elements[natoms_monomer:]

    clash = False

    min_ratio = 999.0
    min_info = None

    for i in range(natoms_monomer):

        for j in range(natoms_monomer):

            if (i, j) in skip_pairs:
                continue

            d = np.linalg.norm(mon1[i] - mon2[j])

            cutoff = scale * (vdw[elem1[i]] + vdw[elem2[j]])

            ratio = d / cutoff

            if ratio < min_ratio:
                min_ratio = ratio
                min_info = (i+1, elem1[i],
                            j+1, elem2[j],
                            d, cutoff)

            if d < cutoff:
                clash = True

    if clash:

        rejected += 1

        i, e1, j, e2, d, cutoff = min_info

        log.write(
            f"{xyz:25s}  "
            f"Clash: {e1}{i} -- {e2}{j}   "
            f"d={d:.3f} Å   cutoff={cutoff:.3f} Å\n"
        )

    else:

        accepted += 1

        shutil.copy(path,
                    os.path.join(output_dir, xyz))

log.close()

print("========================================")
print("Accepted :", accepted)
print("Rejected :", rejected)
print("Total    :", accepted + rejected)
print("========================================")
