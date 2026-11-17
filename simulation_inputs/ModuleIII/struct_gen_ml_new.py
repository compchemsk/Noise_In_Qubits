import numpy as np
import os

np.random.seed(12345)
# ============================================================
# USER INPUT
# ============================================================

input_xyz = "TEMPO-wB97XD.xyz"
outdir = "generated_structures"

# O-O distances
OO_DISTANCES = np.concatenate([
    np.arange(2.0, 6.0, 0.2),   # 2.0–5.9 Å
    np.arange(6.0, 11.0, 1.0)   # 6.0–10.0 Å
])

os.makedirs(outdir, exist_ok=True)

# ============================================================
# READ XYZ
# ============================================================

with open(input_xyz) as f:

    natoms = int(f.readline())
    comment = f.readline()

    elements = []
    coords = []

    for line in f:
        sp = line.split()

        elements.append(sp[0])
        coords.append([
            float(sp[1]),
            float(sp[2]),
            float(sp[3])
        ])

coords = np.array(coords)

# ============================================================
# Locate N and O atoms
# ============================================================

N_index = None
O_index = None

for i, e in enumerate(elements):

    if e == "N" and N_index is None:
        N_index = i

    if e == "O" and O_index is None:
        O_index = i

if N_index is None:
    raise RuntimeError("No nitrogen atom found.")

if O_index is None:
    raise RuntimeError("No oxygen atom found.")

# ============================================================
# Geometry utilities
# ============================================================

def normalize(v):
    return v / np.linalg.norm(v)


def rotation_matrix(axis, theta_deg):
    """
    Rodrigues rotation matrix.
    """

    axis = normalize(axis)

    theta = np.deg2rad(theta_deg)

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc = b*c
    ad = a*d
    ac = a*c
    ab = a*b
    bd = b*d
    cd = c*d

    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


def rotate_about_point(points, origin, R):

    return (points-origin) @ R.T + origin


# ============================================================
# Angle
# ============================================================

def angle(a, b, c):
    """
    Angle ABC (degrees)
    """

    ba = a - b
    bc = c - b

    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)

    x = np.clip(np.dot(ba, bc), -1.0, 1.0)

    return np.degrees(np.arccos(x))


# ============================================================
# Dihedral
# ============================================================

def dihedral(p0, p1, p2, p3):
    """
    Dihedral angle (degrees)
    """

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    ang = np.degrees(np.arctan2(y, x))

    if ang < 0:
        ang += 360.0

    return ang

# ============================================================
# Random rotation
# ============================================================

def random_rotation_matrix():
    """
    Generate a uniformly distributed random rotation matrix.
    (Shoemake, Graphics Gems III)
    """

    u1, u2, u3 = np.random.rand(3)

    q1 = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)   * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)   * np.cos(2*np.pi*u3)

    return np.array([
        [1-2*(q3*q3+q4*q4),   2*(q2*q3-q1*q4),   2*(q2*q4+q1*q3)],
        [2*(q2*q3+q1*q4),     1-2*(q2*q2+q4*q4), 2*(q3*q4-q1*q2)],
        [2*(q2*q4-q1*q3),     2*(q3*q4+q1*q2),   1-2*(q2*q2+q3*q3)]
    ])


# ============================================================
# Fibonacci sphere
# ============================================================

def fibonacci_sphere(n):

    points = []

    golden = np.pi * (3.0 - np.sqrt(5.0))

    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])

    for i in range(n):

        y = 1.0 - (2.0 * i) / (n - 1)

        r = np.sqrt(max(0.0, 1.0 - y*y))

        theta = golden * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        points.append([x, y, z])

    return np.array(points)


# ============================================================
# Reference monomer
# ============================================================

mono1 = coords.copy()

O1 = mono1[O_index]
N1 = mono1[N_index]

counter = 0

# ============================================================
# MAIN LOOP
# ============================================================

for dist in OO_DISTANCES:

    if dist < 6.0:
        npoints = 500
    else:
        npoints = 50

    sphere_points = fibonacci_sphere(npoints)

    for p in sphere_points:

        mono2 = coords.copy()

        # ----------------------------------------------------
        # Place O atom on sphere
        # ----------------------------------------------------

        O2 = mono2[O_index]

        target = O1 + dist * p

        mono2 += target - O2

       # ----------------------------------------------------
       # Give second TEMPO a random orientation
       # ----------------------------------------------------

        O2 = mono2[O_index]

        R = random_rotation_matrix()

        mono2 = rotate_about_point(
            mono2,
            O2,
            R
        )
        # ----------------------------------------------------
        # Compute geometric parameters
        # ----------------------------------------------------

        O2 = mono2[O_index]
        N2 = mono2[N_index]

        OO = np.linalg.norm(O2 - O1)

        ang = angle(O1, O2, N2)

        dih = dihedral(N1, O1, O2, N2)

        # ----------------------------------------------------
        # Write XYZ
        # ----------------------------------------------------

        allatoms = np.vstack((mono1, mono2))
        allelements = elements + elements

        filename = f"struct_{counter:06d}.xyz"

        with open(os.path.join(outdir, filename), "w") as f:

            f.write(f"{2*natoms}\n")

            f.write(
                f"OO={OO:.2f} "
                f"Angle={ang:.2f} "
                f"Dihedral={dih:.2f}\n"
            )

            for e, c in zip(allelements, allatoms):

                f.write(
                    f"{e:2s}"
                    f"{c[0]:15.6f}"
                    f"{c[1]:15.6f}"
                    f"{c[2]:15.6f}\n"
                )

        counter += 1

print(f"\n{counter} structures written.")
