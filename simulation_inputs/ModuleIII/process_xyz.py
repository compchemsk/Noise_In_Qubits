#!/usr/bin/env python3

import glob

for xyzfile in glob.glob("*.xyz"):

    with open(xyzfile, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    comment = lines[1]
    atoms = lines[2:]

    # Find all oxygen atoms
    O_atoms = [line for line in atoms if line.split()[0] == "O"]
    other_atoms = [line for line in atoms if line.split()[0] != "O"]

    if len(O_atoms) != 2:
        print(f"Skipping {xyzfile}: found {len(O_atoms)} oxygen atoms.")
        continue

    new_atoms = O_atoms + other_atoms

    with open(xyzfile, "w") as f:
        f.write(f"{natoms}\n")
        f.write(comment)
        f.writelines(new_atoms)

    print(f"Processed {xyzfile}")
