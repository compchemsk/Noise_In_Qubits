import os
import glob

xyz_dir = "filtered_structures"
output_file = "structure_summary.txt"

files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))

with open(output_file, "w") as fout:

    fout.write("#File\tOO\tAngle\tDihedral\n")

    for file in files:

        with open(file) as f:
            f.readline()                 # Skip first line (number of atoms)
            second_line = f.readline().strip()

        fout.write(f"{os.path.basename(file)}\t{second_line}\n")

print(f"Summary written to {output_file}")
