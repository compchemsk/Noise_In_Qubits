# -*- coding: utf-8 -*-
"""spin_phonon_scans_notebook.py

Original analysis notebook (exported from Google Colab) containing the computed
g_iso(Q) and A_iso(Q) values of every displacement scan. It is read, without
re-typing any number, by python_codes/m1_01_extract_scans.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

"""**mode1 TEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061945

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062086, 2.0062042, 2.0062006, 2.0061977,
              2.0061957, 2.0061945, 2.0061942, 2.0061946,
              2.0061959, 2.0061980, 2.0062009])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 47.4837

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([48.3112,48.1324,47.9602,47.7944,47.6357,47.4837,47.3387,47.2013,47.0711,46.9487,46.8337])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode2 TEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061945

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061920,2.0061928,2.0061935,2.0061940,2.0061943,2.0061945,2.0061946,2.0061945,2.0061943,2.0061939,2.0061934])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 47.4837

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([47.5455,47.5249,47.5085,47.4961,47.4881,47.4837,47.4838,47.4873,47.4951,47.5068,47.5224])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode3 TEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061945

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061658,2.0061649,2.0061673,2.0061730,2.0061821,2.0061945,2.0062105,2.0062299,2.0062530,2.0062796,2.0063100])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.4, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 47.4837

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([44.9853,45.4500,45.9342,46.4362,46.9532,47.4837,48.0251,48.5739,49.1286,49.6857,50.2424])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode1-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0059563,2.0059967,2.0060373,2.0060784,2.0061197,2.0061613,2.0062032,2.0062455,2.0062881,2.0063309,2.0063741])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([45.7075,45.9257,46.1420,46.3460,46.5585,46.7695,46.9804,47.1885,47.3946,47.6011,47.8068])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode2-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0057551,2.0058314,2.0059100,2.0059911,2.0060749,2.0061613,2.0062505,2.0063426,2.0064377,2.0065360,2.0066376])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([43.4361,44.1369,44.8333,45.4991,46.1450,46.7695,47.3721,47.9642,48.5180,49.0454,49.5510])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode3-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061693,2.0061675,2.0061658,2.0061642,2.0061627,2.0061613,2.0061600,2.0061588,2.0061577,2.0061565,2.0061556])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.4, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([47.6604,47.4816,47.3028,47.1243,46.9472,46.7695,46.5927,46.4161,46.2400,46.0758,45.8891])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.9,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode4-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0059356,2.0059792,2.0060236,2.0060687,2.0061146,2.0061613,2.0062088,2.0062570,2.0063061,2.0063559,2.0064066])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([44.6270,45.0747,45.5133,45.9425,46.3613,46.7695,47.1668,47.5524,47.9255,48.2987,48.6460])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode5-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061836,2.0061795,2.0061752,2.0061707,2.0061661,2.0061613,2.0061564,2.0061513,2.0061461,2.0061407,2.0061352])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.5385,46.5774,46.6200,46.6668,46.7166,46.7695,46.8260,46.8862,46.9493,47.0153,47.0850])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.6, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode6-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061521,2.0061540,2.0061559,2.0061577,2.0061595,2.0061613,2.0061630,2.0061648,2.0061664,2.0061681,2.0061697])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.7359,46.7419,46.7489,46.7554,46.7621,46.7695,46.7769,46.7846,46.7922,46.8000,46.8079])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.6, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode7-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0055751,2.0056869,2.0058012,2.0059183,2.0060382,2.0061613,2.0062878,2.0064178,2.0065519,2.0066901,2.0068329])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.2, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([48.4520,48.0622,47.7175,47.3843,47.0691,46.7695,46.4833,46.2204,45.9365,45.6915,45.4347])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**mode8-PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0061613

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0061756,2.0061723,2.0061692,2.0061663,2.0061637,2.0061613,2.0061591,2.0061572,2.0061555,2.0061540,2.0061527])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.4, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.7695

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.3897,46.4723,46.5522,46.6283,46.7005,46.7695,46.8349,46.8963,46.9541,47.0082,47.0586])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.5, 0.4,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Electron Spin Relaxation and Dephasing**"""

!pip install qutip

import numpy as np
from qutip import Bloch

# Create Bloch sphere
b = Bloch()

# Bloch vectors
ket0 = [0, 0, 1]       # |0>
ket1 = [0, 0, -1]      # |1>
half_ket0 = [0, 0, 0.5]
half_ket1 = [0, 0, -0.5]

# Add vectors (no styling here)
b.add_vectors([ket0, ket1, half_ket0, half_ket1])

# Style vectors AFTER adding them
b.vector_color = [
    'blue',   # |0>
    'red',    # |1>
    'black',  # half of |0>
    'green'   # half of |1>
]

b.vector_width = 3

# Show Bloch sphere
b.show()

import numpy as np
from qutip import Bloch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Bloch sphere
b = Bloch()

# Fixed polar angle (equator → max coherence)
theta = np.pi / 2

# Phases representing dephasing
phis = np.linspace(0, np.pi/2, 30, endpoint=False)

# Generate Bloch vectors
vectors = []
for phi in phis:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors.append([x, y, z])

b.add_vectors(vectors)

# Create light → dark color gradient
cmap = cm.jet          # try: Reds, Greens, viridis, plasma
norm = mcolors.Normalize(vmin=0, vmax=len(vectors)-1)
colors = [cmap(norm(i)) for i in range(len(vectors))]

# Apply styling
b.vector_color = colors
b.vector_width = 2

b.show()

"""**Functional Benchmark PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([2.0057903, 2.0061612, 2.0063078, 2.0068677])
colors = ['black', 'red', 'green', 'orange']
labels = ['B3LYP', 'B2PLYP', 'DSD-PBEP86', 'RI-MP2']

plt.bar(x, y, width=0.8, color=colors, label=labels)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Functionals", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')

plt.axhline(y=2.0063, color='blue', linestyle='--')
plt.legend(loc='upper left', frameon=False, fontsize=18)
plt.xticks(x, x, fontsize=18)
plt.yticks(fontsize=18)

plt.xlim(0.5, 4.5)
plt.ylim(2.004,2.008)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([10.83, -3.12, -8.55, -18.82])
colors = ['black', 'red', 'green', 'orange']
labels = ['B3LYP', 'B2PLYP', 'DSD-PBEP86', 'RI-MP2']

plt.bar(x, y, width=0.8, color=colors, label=labels)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Functionals", fontsize=24, fontweight='bold')
plt.ylabel("$\Delta A_{iso}$", fontsize=24, fontweight='bold')

plt.axhline(y=0, color='blue', linestyle='--')
plt.legend(loc='upper right', frameon=False, fontsize=18)
plt.xticks(x, x, fontsize=18)
plt.yticks(fontsize=18)

plt.xlim(0.5, 4.5)
plt.ylim(-20,20)
plt.show()

"""**Atom-wise contribution in TEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
data = np.loadtxt('vibcontrib.dat')

modes = data[0:5, 0]

columns = {
    5: ("NO contribution", "red"),
    6: ("Ring contribution", "green"),
    7: ("Meth$_{H}$ contribution", "orange"),
    8: ("Meth$_{C}$ contribution", "purple"),
}

y1 = data[0:5, 4]
# -----------------------
# Loop over columns
# -----------------------
for col, (ylabel, color) in columns.items():

    y = data[0:5, col]

    plt.figure(figsize=(6, 4))
    plt.bar(modes, y*100, width=0.8, color=color)

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
    plt.ylabel(ylabel, fontsize=22, fontweight='bold')

    plt.xticks(np.arange(0, 6, 1), fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(0, 6)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 4))
plt.bar(modes, y1, width=0.8, color='black')

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"$\sqrt{\hbar / m\omega}$ (cm)", fontsize=22, fontweight='bold')

plt.xticks(np.arange(0, 6, 1), fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 6)

plt.tight_layout()
plt.show()

"""**Wigner Sampling TEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data = np.loadtxt('Wigner_TEMPO_intcoord.dat', skiprows=1)

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("N-O-R1 angle", "green"),
    3: ("N-O-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # Plot normalized histogram
    plt.hist(data[:, col], bins=20, color=color, edgecolor='black', density=True, label=label)

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')  # Fixed ylabel

    # Legend
    plt.legend(fontsize=18)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

"""**atom-wise contribution PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
data = np.loadtxt('vibcontrib_ptempo.dat')

modes = data[0:8, 0]

columns = {
    5: ("NO contribution", "red"),
    6: ("Ring contribution", "green"),
    7: ("Meth$_{H}$ contribution", "orange"),
    8: ("Meth$_{C}$ contribution", "purple"),
    9: ("Prop. contribution", "brown")
}

y1 = data[0:8, 4]
# -----------------------
# Loop over columns
# -----------------------
for col, (ylabel, color) in columns.items():

    y = data[0:8, col]

    plt.figure(figsize=(6, 4))
    plt.bar(modes, y*100, width=0.8, color=color)

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
    plt.ylabel(ylabel, fontsize=22, fontweight='bold')

    plt.xticks(np.arange(0, 9, 1), fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(0, 9)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 4))
plt.bar(modes, y1, width=0.8, color='black')

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"$\sqrt{\hbar / m\omega}$ (cm)", fontsize=22, fontweight='bold')

plt.xticks(np.arange(0, 9, 1), fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 9)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load first dataset
# -----------------------
data1 = np.loadtxt('vibcontrib_ptempo.dat')
modes1 = data1[0:8, 0]
y1 = data1[0:8, 4]

# -----------------------
# Load second dataset
# -----------------------
data2 = np.loadtxt('vibcontrib.dat')
modes2 = data2[0:8, 0]
y2 = data2[0:8, 4]

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6, 4))

width = 0.35  # bar width

# First dataset
plt.bar(modes1 - width/2, y1, width=width, color='black', label='PTEMPO')

# Second dataset
plt.bar(modes2 + width/2, y2, width=width, color='gray', label='TEMPO')

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"$\sqrt{\hbar / m\omega}$ (cm)", fontsize=22, fontweight='bold')

plt.xticks(np.arange(0, 9, 1), fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 9)

plt.legend(fontsize=18, frameon=False)

plt.tight_layout()
plt.show()

"""**Wigner Sampling PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data = np.loadtxt('Wigner_PTEMPO_intcoord.dat', skiprows=1)

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("N-O-R1 angle", "green"),
    3: ("N-O-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # Plot normalized histogram
    plt.hist(data[:, col], bins=20, color=color, edgecolor='black', density=True, label=label)

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')  # Fixed ylabel

    # Legend
    plt.legend(fontsize=18)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

"""**Spin-Phonon Coupling in TEMPO and PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load first dataset
# -----------------------
modes1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y1 = np.array([1.028e-5, 1.784e-6, 7.178e-5, 0, 0, 0, 0, 0])

# -----------------------
# Load second dataset
# -----------------------
modes2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y2 = np.array([2.089e-4, 4.406e-4, 6.832e-6, 2.355e-4, 2.423e-5, 8.800e-6, 6.274e-4, 1.144e-5])

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6, 4))

width = 0.35  # bar width

# First dataset
plt.bar(modes1 - width/2, y1, width=width, color='black', label='TEMPO')

# Second dataset
plt.bar(modes2 + width/2, y2, width=width, color='gray', label='PTEMPO')

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"Zeeman Spin-Phonon", fontsize=18, fontweight='bold')

plt.xticks(np.arange(0, 9, 1), fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 9)

plt.legend(fontsize=18, frameon=False)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load first dataset
# -----------------------
modes1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y1 = np.array([7.397e-1,5.025e-2,2.644e0,0,0,0,0,0])

# -----------------------
# Load second dataset
# -----------------------
modes2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y2 = np.array([1.048, 3.065, 8.836e-1,2.011,2.736e-1,3.616e-2,1.492,3.347e-1])

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(6, 4))

width = 0.35  # bar width

# First dataset
plt.bar(modes1 - width/2, y1, width=width, color='black', label='TEMPO')

# Second dataset
plt.bar(modes2 + width/2, y2, width=width, color='gray', label='PTEMPO')

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"Hyperfine Spin-Phonon", fontsize=17, fontweight='bold')

plt.xticks(np.arange(0, 9, 1), fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 9)

plt.legend(fontsize=18, frameon=False)

plt.tight_layout()
plt.show()

"""**Net Mode Resolved Coupling**"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
hbar = 1.0545718e-34      # J·s
c = 2.99792458e10        # cm/s

# ============================================================
# Vibrational modes
# ============================================================
modes = np.arange(1, 9)
width = 0.35

# ============================================================
# Frequencies (cm^-1)
# ============================================================
nu_ptempo = np.array([78.3, 117.7, 197.8, 0, 0, 0, 0, 0])
nu_tempo  = np.array([29.8, 56.6, 72.8, 86.6, 116.4, 158.8, 196.7, 207.4])

omega_ptempo = 2 * np.pi * c * nu_ptempo
omega_tempo  = 2 * np.pi * c * nu_tempo

# Masks to avoid zero-frequency modes
mask_ptempo = omega_ptempo > 0
mask_tempo  = omega_tempo > 0

# ============================================================
# Zeeman spin–phonon coupling (∂g / ∂Q)
# ============================================================
dg_dQ_ptempo = np.array([1.028e-5, 1.784e-6, 7.178e-5, 0, 0, 0, 0, 0])
dg_dQ_tempo  = np.array([
    2.089e-4, 4.406e-4, 6.832e-6, 2.355e-4,
    2.423e-5, 8.800e-6, 6.274e-4, 1.144e-5
])

# ============================================================
# Hyperfine spin–phonon coupling (∂A / ∂Q)
# ============================================================
dA_dQ_ptempo = np.array([7.397e-1, 5.025e-2, 2.644e0, 0, 0, 0, 0, 0])
dA_dQ_tempo  = np.array([
    1.048, 3.065, 8.836e-1, 2.011,
    2.736e-1, 3.616e-2, 1.492, 3.347e-1
])

# ============================================================
# Mode-resolved NET spin–phonon coupling
# ============================================================
Lambda_net_ptempo = np.zeros_like(modes, dtype=float)
Lambda_net_tempo  = np.zeros_like(modes, dtype=float)

Lambda_net_ptempo[mask_ptempo] = (
    (dg_dQ_ptempo[mask_ptempo]**2 +
     dA_dQ_ptempo[mask_ptempo]**2)
    / (2 * hbar * omega_ptempo[mask_ptempo])
)

Lambda_net_tempo[mask_tempo] = (
    (dg_dQ_tempo[mask_tempo]**2 +
     dA_dQ_tempo[mask_tempo]**2)
    / (2 * hbar * omega_tempo[mask_tempo])
)

# ============================================================
# Plot: mode-resolved net spin–phonon coupling
# ============================================================
plt.figure(figsize=(6, 4))

plt.bar(modes - width/2, Lambda_net_ptempo,
        width=width, color='black', label='TEMPO')

plt.bar(modes + width/2, Lambda_net_tempo,
        width=width, color='gray', label='PTEMPO')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.xlabel("Vibrational Modes", fontsize=22, fontweight='bold')
plt.ylabel(r"Net Spin–Phonon Coupling    ",
           fontsize=18, fontweight='bold')

plt.xticks(modes, fontsize=20)
plt.yticks(fontsize=18)
plt.xlim(0, 9)

plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.show()

"""**TEMPO-SAM**

**Spin Crowding**
"""

!pip install MDAnalysis

#!/usr/bin/env python3
"""
Time-resolved spin crowding analysis (WSCI + fluctuations)

Spin centers   : name O1
Reference spin : resid 2486 and name O1
BOX-INDEPENDENT (safe for XTC without unit cell)
"""

# ====================================================
# 1. IMPORTS
# ====================================================
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# 2. LOAD SYSTEM
# ====================================================
u = mda.Universe("TEMPO-SAM-11-300.parm7", "TEMPO-SAM-11-300.xtc")

spin_centers = u.select_atoms("name O1")
ref_spin = u.select_atoms("resid 2486 and name O1")

if len(ref_spin) != 1:
    raise ValueError("Reference spin center must be unique")

print(f"Total spin centers: {len(spin_centers)}")
print("Box dimensions (first frame):", u.dimensions)

# ====================================================
# 3. DETERMINE DECAY LENGTH (λ) FROM DISTANCE PROFILE
# ====================================================
all_distances = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]
    d = d[d > 1e-3]
    all_distances.extend(d)

all_distances = np.array(all_distances)

bins = np.linspace(0.0, 15.0, 300)
counts, edges = np.histogram(all_distances, bins=bins, density=False)
r = 0.5 * (edges[:-1] + edges[1:])

gr_tilde = counts / (r**2 + 1e-6)   # geometry-corrected
gr_tilde /= gr_tilde.max()          # normalize for visualization
lambda_decay = r[np.argmax(gr_tilde)]
print(f"Corrected λ = {lambda_decay:.2f} Å")

# ====================================================
# 4. WSCI (TIME-RESOLVED)
# ====================================================
WSCI = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]

    d = d[d > 1e-3]
    wsci = np.sum(np.exp(-d / lambda_decay))
    WSCI.append(wsci)

WSCI = np.array(WSCI)

# ====================================================
# 5. TIME-RESOLVED CROWDING FLUCTUATION
# ====================================================
WSCI_mean = WSCI.mean()
delta_WSCI = WSCI - WSCI_mean

print("\nCrowding statistics:")
print(f"<WSCI> = {WSCI_mean:.3f}")
print(f"σ(WSCI) = {WSCI.std():.3f}")

# ====================================================
# 6. PLOTTING STYLE
# ====================================================
def style_axis(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# ====================================================
# 7. FINAL PLOT (ONLY ONE)
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))
x1 = np.arange(len(WSCI))
ax.plot(x1/100,WSCI, lw=2.5, label="WSCI", color='blue')
ax.plot(x1/100,delta_WSCI, lw=2.5, label=r"$\Delta$WSCI", color='red')

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Crowding Index")
ax.set_xlim(0, 15)
ax.set_ylim(-0.5,2.5)
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

# ====================================================
# 8. SAVE DATA
# ====================================================
np.savetxt(
    "wsci_time_series.dat",
    np.column_stack([WSCI, delta_WSCI]),
    header="WSCI  Delta_WSCI"
)

print("\nAnalysis complete.")
print("Saved: wsci_time_series.dat")
# ====================================================
# 3b. PLOT DISTANCE PROFILE TO SHOW λ
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(r, hist, lw=3, label="Spin–spin distance profile")
ax.axvline(lambda_decay, lw=2.5, ls="--",
           label=rf"$\lambda = {lambda_decay:.2f}\ \mathrm{{\AA}}$",
           color="black")

ax.set_xlabel("r (Å)")
ax.set_ylabel("Probability Density")
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

!pip install MDAnalysis -q

#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ====================================================
# INPUT FILES
# ====================================================
systems = [
    ("300K", "TEMPO-SAM-11-300K.parm7", "TEMPO-SAM-11-300K.xtc"),
    ("100K", "TEMPO-SAM-11-100K.parm7", "TEMPO-SAM-11-100K.xtc"),
]

radicals = [
    ("Central TEMPO", "resid 2486 and name O1"),
    ("Edge TEMPO", "resid 2426 and name O1"),
]

colors = ["black", "blue", "red", "green"]

# ====================================================
# FUNCTION: COMPUTE WSCI (UNCHANGED PHYSICS)
# ====================================================
def compute_wsci(u, ref_sel):
    spin_centers = u.select_atoms("name O1")
    ref_spin = u.select_atoms(ref_sel)

    if len(ref_spin) != 1:
        raise ValueError(f"{ref_sel} not unique")

    # ---- λ calculation ----
    all_distances = []

    for ts in u.trajectory[-500:]:
        d = mda.lib.distances.distance_array(
            ref_spin.positions,
            spin_centers.positions,
            box=None
        )[0]
        d = d[d > 1e-3]
        all_distances.extend(d)

    all_distances = np.array(all_distances)

    bins = np.linspace(0.0, 15.0, 300)
    counts, edges = np.histogram(all_distances, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])

    gr_tilde = counts / (r**2 + 1e-6)
    gr_tilde /= gr_tilde.max()

    lambda_decay = r[np.argmax(gr_tilde)]

    # ---- WSCI ----
    WSCI = []

    for ts in u.trajectory[-500:]:
        d = mda.lib.distances.distance_array(
            ref_spin.positions,
            spin_centers.positions,
            box=None
        )[0]

        d = d[d > 1e-3]
        wsci = np.sum(np.exp(-d / lambda_decay))
        WSCI.append(wsci)

    return np.array(WSCI)

# ====================================================
# COLLECT ALL DATA
# ====================================================
all_wsci = []
labels = []

for sys_name, top, traj in systems:
    u = mda.Universe(top, traj)

    for rad_name, sel in radicals:
        print(f"Processing {sys_name} - {rad_name}")
        wsci = compute_wsci(u, sel)

        all_wsci.append(wsci)
        labels.append(f"{sys_name}-{rad_name}")

# ====================================================
# KDE PLOT (ALL 4 TOGETHER)
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))

for wsci, label, color in zip(all_wsci, labels, colors):

    # Histogram → smooth
    bins = np.linspace(wsci.min(), wsci.max(), 200)
    counts, edges = np.histogram(wsci, bins=bins, density=True)

    x = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts, sigma=2)

    # ====================================================
    # NORMALIZE TO 0–1
    # ====================================================
    smooth /= smooth.max()

    mean = wsci.mean()
    std = wsci.std()

    # Curve
    ax.plot(x, smooth, lw=2.5, color=color, label=label)

    # ====================================================
    # FULL AREA SHADING
    # ====================================================
    ax.fill_between(
        x,
        0,
        smooth,
        color=color,
        alpha=0.25
    )

    # Mean line
    ax.axvline(mean, color=color, lw=2, ls="--")

# ====================================================
# STYLE
# ====================================================
ax.set_xlabel(r"$\eta_{SC}$", fontsize=26, fontweight="bold")
ax.set_ylabel("Normalized Probability", fontsize=24, fontweight="bold")

ax.set_ylim(0, 1.05)

ax.tick_params(axis="both", labelsize=20, width=2)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.legend(fontsize=18, frameon=False, loc='lower right')

plt.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Time-resolved spin crowding analysis (WSCI + fluctuations)

Spin centers   : name O1
Reference spin : resid 2486 and name O1
BOX-INDEPENDENT (safe for XTC without unit cell)
"""

# ====================================================
# 1. IMPORTS
# ====================================================
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# 2. LOAD SYSTEM
# ====================================================
u = mda.Universe("TEMPO-SAM-11-100.parm7", "TEMPO-SAM-11-100.xtc")

spin_centers = u.select_atoms("name O1")
ref_spin = u.select_atoms("resid 2426 and name O1")

if len(ref_spin) != 1:
    raise ValueError("Reference spin center must be unique")

print(f"Total spin centers: {len(spin_centers)}")
print("Box dimensions (first frame):", u.dimensions)

# ====================================================
# 3. DETERMINE DECAY LENGTH (λ) FROM DISTANCE PROFILE
# ====================================================
all_distances = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]
    d = d[d > 1e-3]
    all_distances.extend(d)

all_distances = np.array(all_distances)

bins = np.linspace(0.0, 15.0, 300)
counts, edges = np.histogram(all_distances, bins=bins, density=False)
r = 0.5 * (edges[:-1] + edges[1:])

gr_tilde = counts / (r**2 + 1e-6)   # geometry-corrected
gr_tilde /= gr_tilde.max()          # normalize for visualization
lambda_decay = r[np.argmax(gr_tilde)]
print(f"Corrected λ = {lambda_decay:.2f} Å")

# ====================================================
# 4. WSCI (TIME-RESOLVED)
# ====================================================
WSCI = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]

    d = d[d > 1e-3]
    wsci = np.sum(np.exp(-d / lambda_decay))
    WSCI.append(wsci)

WSCI = np.array(WSCI)

# ====================================================
# 5. TIME-RESOLVED CROWDING FLUCTUATION
# ====================================================
WSCI_mean = WSCI.mean()
delta_WSCI = WSCI - WSCI_mean

print("\nCrowding statistics:")
print(f"<WSCI> = {WSCI_mean:.3f}")
print(f"σ(WSCI) = {WSCI.std():.3f}")

# ====================================================
# 6. PLOTTING STYLE
# ====================================================
def style_axis(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# ====================================================
# 7. FINAL PLOT (ONLY ONE)
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))
x1 = np.arange(len(WSCI))
ax.plot(x1/100,WSCI, lw=2.5, label="WSCI", color='blue')
ax.plot(x1/100,delta_WSCI, lw=2.5, label=r"$\Delta$WSCI", color='red')

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Crowding Index")
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

# ====================================================
# 8. SAVE DATA
# ====================================================
np.savetxt(
    "wsci_time_series.dat",
    np.column_stack([WSCI, delta_WSCI]),
    header="WSCI  Delta_WSCI"
)

print("\nAnalysis complete.")
print("Saved: wsci_time_series.dat")
# ====================================================
# 3b. PLOT DISTANCE PROFILE TO SHOW λ
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(r, hist, lw=3, label="Spin–spin distance profile")
ax.axvline(lambda_decay, lw=2.5, ls="--",
           label=rf"$\lambda = {lambda_decay:.2f}\ \mathrm{{\AA}}$",
           color="black")

ax.set_xlabel("r (Å)")
ax.set_ylabel("Probability Density")
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Time-resolved spin crowding analysis (WSCI + fluctuations)

Spin centers   : name O1
Reference spin : resid 2486 and name O1
BOX-INDEPENDENT (safe for XTC without unit cell)
"""

# ====================================================
# 1. IMPORTS
# ====================================================
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# 2. LOAD SYSTEM
# ====================================================
u = mda.Universe("TEMPO-SAM-11.parm7", "TEMPO-SAM-11.xtc")

spin_centers = u.select_atoms("name O1")
ref_spin = u.select_atoms("resid 2426 and name O1")

if len(ref_spin) != 1:
    raise ValueError("Reference spin center must be unique")

print(f"Total spin centers: {len(spin_centers)}")
print("Box dimensions (first frame):", u.dimensions)

# ====================================================
# 3. DETERMINE DECAY LENGTH (λ) FROM DISTANCE PROFILE
# ====================================================
all_distances = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]
    d = d[d > 1e-3]
    all_distances.extend(d)

all_distances = np.array(all_distances)

bins = np.linspace(0.0, 15.0, 300)
counts, edges = np.histogram(all_distances, bins=bins, density=False)
r = 0.5 * (edges[:-1] + edges[1:])

gr_tilde = counts / (r**2 + 1e-6)   # geometry-corrected
gr_tilde /= gr_tilde.max()          # normalize for visualization
lambda_decay = r[np.argmax(gr_tilde)]
print(f"Corrected λ = {lambda_decay:.2f} Å")

# ====================================================
# 4. WSCI (TIME-RESOLVED)
# ====================================================
WSCI = []

for ts in u.trajectory:
    d = mda.lib.distances.distance_array(
        ref_spin.positions,
        spin_centers.positions,
        box=None
    )[0]

    d = d[d > 1e-3]
    wsci = np.sum(np.exp(-d / lambda_decay))
    WSCI.append(wsci)

WSCI = np.array(WSCI)

# ====================================================
# 5. TIME-RESOLVED CROWDING FLUCTUATION
# ====================================================
WSCI_mean = WSCI.mean()
delta_WSCI = WSCI - WSCI_mean

print("\nCrowding statistics:")
print(f"<WSCI> = {WSCI_mean:.3f}")
print(f"σ(WSCI) = {WSCI.std():.3f}")

# ====================================================
# 6. PLOTTING STYLE
# ====================================================
def style_axis(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=24, fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")

# ====================================================
# 7. FINAL PLOT (ONLY ONE)
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))
x1 = np.arange(len(WSCI))
ax.plot(x1/100,WSCI, lw=2.5, label="WSCI", color='blue')
ax.plot(x1/100,delta_WSCI, lw=2.5, label=r"$\Delta$WSCI", color='red')

ax.set_xlabel("Time (ns)")
ax.set_ylabel("Crowding Index")
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

# ====================================================
# 8. SAVE DATA
# ====================================================
np.savetxt(
    "wsci_time_series.dat",
    np.column_stack([WSCI, delta_WSCI]),
    header="WSCI  Delta_WSCI"
)

print("\nAnalysis complete.")
print("Saved: wsci_time_series.dat")
# ====================================================
# 3b. PLOT DISTANCE PROFILE TO SHOW λ
# ====================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(r, hist, lw=3, label="Spin–spin distance profile")
ax.axvline(lambda_decay, lw=2.5, ls="--",
           label=rf"$\lambda = {lambda_decay:.2f}\ \mathrm{{\AA}}$",
           color="black")

ax.set_xlabel("r (Å)")
ax.set_ylabel("Probability Density")
ax.legend(fontsize=18, frameon=False)

style_axis(ax)
plt.tight_layout()
plt.show()

"""**Vibrational Mode**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-300K_centin_tcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO-SAM-300K_edg_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("N-O-R1 angle", "green"),
    3: ("N-O-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(Central)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.4,
        label=f"{label} (Edge)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=15)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-100K_cent_intcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO-SAM-100K_edg_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("N-O-R1 angle", "green"),
    3: ("N-O-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(Central)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.4,
        label=f"{label} (Edge)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=15)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

"""**PTEMPO vs TEMPO-SAM**"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-300K_centin_tcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO-SAM-100K_cent_intcoord.dat', skiprows=1)  # <-- second dataset
data3 = np.loadtxt('Wigner_PTEMPO_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("O-N-R1 angle", "green"),
    3: ("O-N-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(300K)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.6,
        label=f"{label} (100K)"
    )

    plt.hist(
        data3[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.3,
        label=f"{label} (PTEMPO)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=10)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-300K_edg_intcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO-SAM-100K_edg_intcoord.dat', skiprows=1)  # <-- second dataset
data3 = np.loadtxt('Wigner_PTEMPO_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("O-N-R1 angle", "green"),
    3: ("O-N-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(300K)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.6,
        label=f"{label} (100K)"
    )

    plt.hist(
        data3[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.3,
        label=f"{label} (PTEMPO)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=10)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-300K_centin_tcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO_intcoord.dat', skiprows=1)  # <-- second dataset
data3 = np.loadtxt('Wigner_PTEMPO_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("O-N-R1 angle", "green"),
    3: ("O-N-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(TEMPO)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data3[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.6,
        label=f"{label} (PTEMPO)"
    )

    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.3,
        label=f"{label} (TEMPO-SAM)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=10)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data (skip header)
# -----------------------
data1 = np.loadtxt('Wigner_TEMPO-SAM-100K_cent_intcoord.dat', skiprows=1)
data2 = np.loadtxt('Wigner_TEMPO_intcoord.dat', skiprows=1)  # <-- second dataset
data3 = np.loadtxt('Wigner_PTEMPO_intcoord.dat', skiprows=1)  # <-- second dataset

# Column indices and legend labels
columns = {
    1: ("N-O bond", "red"),
    2: ("O-N-R1 angle", "green"),
    3: ("O-N-R2 angle", "blue"),
    4: ("N pyramidalization", "black"),
    5: ("Dihedral M1-R1-N-O", "brown")
}

# -----------------------
# Plot distribution for each column
# -----------------------
for col, (label, color) in columns.items():
    plt.figure(figsize=(6, 4))

    # First dataset (dark)
    plt.hist(
        data2[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.9,
        label=f"{label}(TEMPO)"
    )

    # Second dataset (lighter shade)
    plt.hist(
        data3[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.6,
        label=f"{label} (PTEMPO)"
    )

    plt.hist(
        data1[:, col],
        bins=20,
        color=color,
        edgecolor='black',
        density=True,
        alpha=0.3,
        label=f"{label} (TEMPO-SAM)"
    )

    # Axis styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    plt.xlabel("Values", fontsize=22, fontweight='bold')
    plt.ylabel("counts", fontsize=22, fontweight='bold')

    # Legend
    plt.legend(fontsize=10)

    # Font sizes and rotation
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    plt.show()

import numpy as np
import MDAnalysis as mda

# --------------------------------------------
# USER SETTINGS
topology_file = "TEMPO-SAM-11-100.parm7"
trajectory_file = "TEMPO-SAM-11-100.xtc"

resid = 2426
atom_N = "N1"
atom_O = "O1"

temperature_K = 300.0   # <-- SET YOUR SIMULATION TEMPERATURE

start_frame = 1
end_frame = 1100        # inclusive
# --------------------------------------------


def extract_bond_length(universe, resid, atom1, atom2, start, end):
    """
    Extract N–O bond length (in meters) for selected frames.
    """
    sel_N = universe.select_atoms(f"resid {resid} and name {atom1}")
    sel_O = universe.select_atoms(f"resid {resid} and name {atom2}")

    if len(sel_N) != 1 or len(sel_O) != 1:
        raise ValueError("Atom selection is not unique.")

    bond_lengths = []

    for ts in universe.trajectory[start:end + 1]:
        r_angstrom = np.linalg.norm(
            sel_O.positions[0] - sel_N.positions[0]
        )
        bond_lengths.append(r_angstrom * 1e-10)  # Å → m

    return np.array(bond_lengths)


# ---------------- MAIN SCRIPT ----------------

# Physical constant
kB = 1.380649e-23  # J/K

u = mda.Universe(topology_file, trajectory_file)

bond_lengths = extract_bond_length(
    u, resid, atom_N, atom_O, start_frame, end_frame
)

r_mean = np.mean(bond_lengths)
variance = np.mean((bond_lengths - r_mean)**2)

k_force = kB * temperature_K / variance   # N/m

print("N–O force constant from MD (harmonic assumption)")
print("------------------------------------------------")
print(f"Frames used      : {start_frame}–{end_frame}")
print(f"Mean bond length : {r_mean*1e10:.4f} Å")
print(f"⟨(Δr)²⟩          : {variance:.3e} m²")
print(f"Force constant k : {k_force:.1f} N/m")

# Reduced mass
amu = 1.66053906660e-27
m_N = 14.0067 * amu
m_O = 15.999 * amu
mu = (m_N * m_O) / (m_N + m_O)

# Speed of light (cm/s)
c = 2.99792458e10

nu_cm = (1.0 / (2.0 * np.pi * c)) * np.sqrt(k_force / mu)
print(f"N–O stretching frequency ≈ {nu_cm:.1f} cm⁻¹")

"""**Relaxation time model for PTEMPO**"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# ================================
# Experimental data
# ================================
T = np.array([7, 10, 20, 30, 40, 50, 60], dtype=float)

# T1 in ms → seconds
T1 = np.array([82, 57, 25.7, 13.0, 4.2, 2.0, 1.0], dtype=float) * 1e-3
err_T1 = np.array([2, 1, 6, 4, 1, 7, 4], dtype=float) * 1e-3

R_exp = 1 / T1
err_R = err_T1 / T1**2

# ================================
# Physical constants
# ================================
hbar = 1.054571817e-34
kB   = 1.380649e-23
c    = 2.99792458e10  # cm/s

# ================================
# X-band frequency
# ================================
nu_1 = 9.5e9
omega_1 = 2 * np.pi * nu_1

# ================================
# Phonon modes
# ================================
omega_sp_cm = np.array([29.8, 56.7, 72.8, 86.6, 116.4, 158.8, 196.7, 207.4])
omega_sp = 2 * np.pi * c * omega_sp_cm

Vsp = np.array([
    2.089e-4, 4.406e-4, 6.832e-6, 2.355e-4,
    2.423e-5, 8.800e-6, 6.274e-4, 1.144e-5
])

# ================================
# Relaxation model
# ================================
def direct_process(T, A):
    x = hbar * omega_1 / (kB * T)
    return A * np.exp(x) / (np.exp(x) - 1)

def phonon_process(T, C):
    total = np.zeros_like(T)
    for w, v in zip(omega_sp, Vsp):
        x = hbar * w / (kB * T)
        total += v * np.exp(x) / (np.exp(x) - 1)**2
    return C * total

def relaxation_rate(T, A, B, C):
    return direct_process(T, A) + B + phonon_process(T, C)

# ================================
# Weighted nonlinear fit
# ================================
p0 = [1e2, 1.0, 1e4]
bounds = (0, np.inf)

popt, pcov = curve_fit(
    relaxation_rate,
    T,
    R_exp,
    sigma=err_R,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds
)

A_fit, B_fit, C_fit = popt

# ================================
# Temperature grid for plotting
# ================================
T_fit = np.linspace(6, 65, 500)

R_direct = direct_process(T_fit, A_fit)
R_phonon = phonon_process(T_fit, C_fit)
R_resid  = B_fit * np.ones_like(T_fit)
R_total  = R_direct + R_phonon + R_resid

# ================================
# Plot styling
# ================================
plt.figure(figsize=(9, 6))

# Bright colors
c_exp = "black"
c_tot = "#e41a1c"   # red
c_dir = "#377eb8"   # blue
c_ph  = "#4daf4a"   # green
c_res = "#ff7f00"   # orange

# Experimental data
plt.errorbar(
    T, R_exp,
    fmt='o', color=c_exp, markersize=7,
    label='Experiment'
)

# Fits
plt.plot(T_fit, R_total, color=c_tot, lw=3, label=' Experimental')
plt.plot(T_fit, R_direct, '--', color=c_dir, lw=2.5, label='Direct')
plt.plot(T_fit, R_phonon, '-.', color=c_ph, lw=2.5, label='Phonon')
plt.plot(T_fit, R_resid, ':', color=c_res, lw=2.5, label='Residual')

# Temperature regions
plt.axvspan(6, 15, color='#a6cee3', alpha=0.5)
plt.axvspan(15, 35, color='#b2df8a', alpha=0.5)
plt.axvspan(35, 65, color='#fdbf6f', alpha=0.5)

# Labels & ticks
plt.xlabel("Temperature (K)", fontsize=24, fontweight='bold')
plt.ylabel(r"$1/T_1$ in log scale", fontsize=24, fontweight='bold')

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.yscale('log')

plt.legend(fontsize=18, frameon=False)
plt.tight_layout()

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

plt.show()

# ================================
# Range-wise contribution analysis
# ================================
T_ranges = {
    "Low (7–15 K)": (7, 15),
    "Mid (15–35 K)": (15, 35),
    "High (35–65 K)": (35, 65)
}

rows = []

for label, (Tmin, Tmax) in T_ranges.items():

    T_grid = np.linspace(Tmin, Tmax, 400)

    R_d = direct_process(T_grid, A_fit)
    R_p = phonon_process(T_grid, C_fit)
    R_b = B_fit * np.ones_like(T_grid)

    R_d_avg = np.mean(R_d)
    R_p_avg = np.mean(R_p)
    R_b_avg = np.mean(R_b)
    R_tot   = R_d_avg + R_p_avg + R_b_avg

    rows.append({
        "Temperature range": label,
        "Direct (s⁻¹)": R_d_avg,
        "Phonon (s⁻¹)": R_p_avg,
        "Residual (s⁻¹)": R_b_avg,
        "Total (s⁻¹)": R_tot,
        "Direct (%)": 100 * R_d_avg / R_tot,
        "Phonon (%)": 100 * R_p_avg / R_tot,
        "Residual (%)": 100 * R_b_avg / R_tot
    })

range_table = pd.DataFrame(rows)

print("\nRange-wise average relaxation contributions:")
print(range_table.round(3))

"""TEMPO SAM -11

**Mode 1**
"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0063088, 2.0062960, 2.0062726,
              2.0062619, 2.0062519,2.0062426, 2.0062339,
              2.0062260, 2.0062187, 2.0062120])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.50, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([45.6849, 45.7664, 45.9457, 46.0437, 46.1472, 46.2565, 46.3716, 46.4924, 46.6191, 46.7517])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 6**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.6, 0.8, 1.0])

y = np.array([2.0062693, 2.0062655, 2.0062619,
              2.0062584, 2.0062551, 2.0062519, 2.0062489,
              2.0062433, 2.0062407, 2.0062383])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.50, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1471

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.6, 0.8, 1.0])

y = np.array([46.2111, 46.1951, 46.1792,
              46.1664, 46.1559, 46.1472, 46.1408,
              46.1341, 46.1341, 46.1362])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 7**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062248, 2.0062298, 2.0062350,
              2.0062404, 2.0062461,2.0062519, 2.0062579,
              2.0062642, 2.0062707, 2.0062773, 2.0062842])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.20, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.1626, 46.1567, 46.1523, 46.1492, 46.1475, 46.1472, 46.1484, 46.1507, 46.1546, 46.1598, 46.1663])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 8**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-0.8, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.8])

y = np.array([2.0062640, 2.0062572, 2.0062543, 2.0062519, 2.0062498,
              2.0062482, 2.0062460])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.40, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-0.8, -0.4, -0.2, 0.0,
              0.2, 0.4, 0.8])

y = np.array([46.1049, 46.1192, 46.1315, 46.1472, 46.1662, 46.1893, 46.2456])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 9**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.4, 0.8, 1.0])

y = np.array([2.0062329, 2.0062364, 2.0062400,
              2.0062438, 2.0062478, 2.0062519, 2.0062606,
              2.0062700, 2.0062749])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.20, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.4, 0.8, 1.0])

y = np.array([46.0693, 46.0796, 46.0944, 46.1115, 46.1281, 46.1470, 46.1895, 46.2373, 46.2631])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 10**

*Seems erroneous*
"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.4, -0.2, 0.0, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062818, 2.0062754, 2.0062632,
              2.0062391, 2.0062519, 2.0062413,
              2.0062363, 2.0062315, 2.0062269])


plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.50, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1480

x = np.array([-1.0, -0.8, -0.4, -0.2, 0.0, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.1401, 46.1428, 46.1473, 46.5842, 46.1480, 46.1534, 46.1559, 46.1598, 46.1626])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 11**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, 0.0, 0.2, 0.6, 1.0])

y = np.array([2.0062448, 2.0062455, 2.0062465,
              2.0062480, 2.0062519, 2.0062543, 2.0062602,
              2.0062675])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1480

x = np.array([-1.0, -0.8, -0.6, -0.4, 0.0, 0.2, 0.6, 1.0])

y = np.array([46.2908, 46.2588, 46.2336, 46.2003, 46.1479, 46.1271, 46.0885, 46.0600])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.40, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 12**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062522, 2.0062519, 2.0062517,
              2.0062516, 2.0062517, 2.0062519, 2.0062523,
              2.0062528, 2.0062534, 2.0062542, 2.0062552])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([45.8146, 45.8780, 45.9429, 46.0094, 46.0775, 46.1472, 46.2185, 46.2915, 46.3659, 46.4420, 46.5198])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 13**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062928, 2.0062840, 2.0062755, 2.0062673,
              2.0062594, 2.0062519, 2.0062447, 2.0062378,
              2.0062312, 2.0062249, 2.0062190])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.40, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([46.3811, 46.3288, 46.2790, 46.2324,
              46.1884, 46.1472, 46.1089, 46.0734,
              46.0408, 46.0112, 45.9847])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.40, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

"""**Mode 24**"""

#g-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

g0 = 2.0062519

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([2.0062344, 2.0062317, 2.0062320, 2.0062355,
              2.0062421, 2.0062519, 2.0062648, 2.0062810,
              2.0063005, 2.0063232, 2.0063493])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - g0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return g0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = g0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, g0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {g0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.40, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)


plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("g$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()

#A-tensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

A0 = 46.1472

x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

y = np.array([43.6834, 44.1414, 44.6184, 45.1129,
              45.6231, 46.1472, 46.6832, 47.2288,
              47.7819, 48.3397, 48.9001])

plt.scatter(x, y, color='black', marker='*')

# Shift data by g0
y1 = y - A0

# Quadratic model without constant term (since g0 is removed)
def func(x, a, b):
    return A0 + a * x + b * x**2

# Design matrix
A = np.column_stack([x, x**2])

# Least-squares fit
sol, _, _, _ = np.linalg.lstsq(A, y1, rcond=None)
firstorder, secondorder = sol

# Fitted values
fit = A0 + firstorder * x + secondorder * x**2

# Correct R² calculation
R2 = r2_score(y1, firstorder * x + secondorder * x**2)
print("R² =", R2)

# Smooth curve for plotting
xf = np.linspace(x.min(), x.max(), 300)
plt.plot(xf, A0 + firstorder * xf + secondorder * xf**2, color = 'blue', linestyle='--', linewidth=2)

# Axis styling
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2)

textstr = (
    f"$c_0 = {A0:.7f}$\n"
    f"$c_1 = {firstorder:.3e}$\n"
    f"$c_2 = {secondorder:.3e}$\n"
    f"$R^2 = {R2:.2f}$"
)

ax.text(
    0.30, 0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=15,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
)

plt.xlabel("Q", fontsize=24, fontweight='bold')
plt.ylabel("A$_{iso}$", fontsize=24, fontweight='bold')
plt.xticks(fontsize=18,rotation=45)
plt.yticks(fontsize=18)
plt.show()

!pip install MDAnalysis -q

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from numpy.linalg import eigh


# ============================================================
# INPUT
# ============================================================

PARM = "TEMPO-SAM-11-100.parm7"
TRAJ = "TEMPO-SAM-11-100.xtc"

RESNAME = "LIN"
RESID = 2426


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):

    ba = a - b
    bc = c - b

    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)

    val = np.clip(np.dot(ba, bc), -1.0, 1.0)

    return np.degrees(np.arccos(val))


def dihedral(p0, p1, p2, p3):

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))


def principal_axis(coords):

    center = coords.mean(axis=0)

    X = coords - center

    cov = X.T @ X

    eigval, eigvec = eigh(cov)

    return eigvec[:, np.argmax(eigval)]


def plane_normal(coords):

    center = coords.mean(axis=0)

    X = coords - center

    cov = X.T @ X

    eigval, eigvec = eigh(cov)

    return eigvec[:, np.argmin(eigval)]


def radius_of_gyration(coords):

    center = coords.mean(axis=0)

    return np.sqrt(
        np.mean(
            np.sum((coords - center)**2, axis=1)
        )
    )


# ============================================================
# LOAD SYSTEM
# ============================================================

u = mda.Universe(PARM, TRAJ)

res = u.select_atoms(
    f"resname {RESNAME} and resid {RESID}"
)

if len(res) == 0:
    raise RuntimeError(
        f"Could not find {RESNAME} {RESID}"
    )


def atom(name):

    ag = res.select_atoms(f"name {name}")

    if len(ag) != 1:
        raise RuntimeError(
            f"Problem with atom {name}: found {len(ag)}"
        )

    return ag[0]


# ============================================================
# ATOMS
# ============================================================

# TEMPO

N1 = atom("N1")
O1 = atom("O1")
C2 = atom("C2")
C3 = atom("C3")
C9 = atom("C9")

# TRIAZOLE

N4 = atom("N4")
N3 = atom("N3")
N2 = atom("N2")
C11 = atom("C11")
C12 = atom("C12")

# LINKER

C10 = atom("C10")
O2 = atom("O2")

# CHAIN

C23 = atom("C23")
C22 = atom("C22")
C21 = atom("C21")
C20 = atom("C20")
C19 = atom("C19")
C18 = atom("C18")
C17 = atom("C17")
C16 = atom("C16")
C15 = atom("C15")
C14 = atom("C14")
C13 = atom("C13")

chain = [
    C23,C22,C21,C20,C19,
    C18,C17,C16,C15,C14,C13
]


# ============================================================
# STORAGE
# ============================================================

data = {


    # TEMPO

    "NO_bond": [],
    "ONO_angle": [],

    "tors1_sin": [],
    "tors1_cos": [],

    "tors2_sin": [],
    "tors2_cos": [],


    # ALKYL

    "chain_Ree": [],
    "chain_Rg": [],
    "chain_bend": [],
    "chain_ring_tilt": [],


    # TRIAZOLE

    "ring_breath": [],
    "ring_asym": [],
    "ring_pucker": [],


    # COUPLING

    "C13_N4_bond": [],
    "C14_C13_N4_angle": [],

    "attach_sin": [],
    "attach_cos": []
}


# ============================================================
# TRAJECTORY LOOP
# ============================================================

for ts in u.trajectory:

    # --------------------------------------------------------
    # TEMPO
    # --------------------------------------------------------

    no_bond = distance(
        N1.position,
        O1.position
    )

    ono_angle = angle(
        O1.position,
        N1.position,
        C3.position
    )

    tors1 = dihedral(
        O1.position,
        N1.position,
        C3.position,
        C9.position
    )

    tors2 = dihedral(
        C3.position,
        N1.position,
        O1.position,
        C2.position
    )

    data["NO_bond"].append(no_bond)
    data["ONO_angle"].append(ono_angle)

    data["tors1_sin"].append(
        np.sin(np.radians(tors1))
    )
    data["tors1_cos"].append(
        np.cos(np.radians(tors1))
    )

    data["tors2_sin"].append(
        np.sin(np.radians(tors2))
    )
    data["tors2_cos"].append(
        np.cos(np.radians(tors2))
    )

    # --------------------------------------------------------
    # CHAIN
    # --------------------------------------------------------

    chain_xyz = np.array(
        [a.position for a in chain]
    )

    Ree = distance(
        C23.position,
        C13.position
    )

    Rg = radius_of_gyration(
        chain_xyz
    )

    bend = angle(
        C23.position,
        C18.position,
        C13.position
    )

    chain_axis = principal_axis(
        chain_xyz
    )

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring_xyz = np.array([

        N4.position,
        N3.position,
        N2.position,
        C11.position,
        C12.position
    ])

    ring_normal = plane_normal(
        ring_xyz
    )

    ring_bonds = [

        distance(N4.position, N3.position),
        distance(N3.position, N2.position),
        distance(N2.position, C11.position),
        distance(C11.position, C12.position),
        distance(C12.position, N4.position)
    ]

    ring_breath = np.mean(
        ring_bonds
    )

    ring_asym = np.std(
        ring_bonds
    )

    center = ring_xyz.mean(axis=0)

    d = np.abs(
        (ring_xyz - center) @ ring_normal
    )

    ring_pucker = np.sqrt(
        np.mean(d**2)
    )

    # --------------------------------------------------------
    # CHAIN-RING TILT
    # --------------------------------------------------------

    tilt = np.degrees(
        np.arccos(
            np.clip(
                np.abs(
                    np.dot(
                        chain_axis,
                        ring_normal
                    )
                ),
                -1,
                1
            )
        )
    )

    # --------------------------------------------------------
    # ATTACHMENT
    # --------------------------------------------------------

    bond_CN = distance(
        C13.position,
        N4.position
    )

    ang_attach = angle(
        C14.position,
        C13.position,
        N4.position
    )

    tors_attach = dihedral(
        C15.position,
        C14.position,
        C13.position,
        N4.position
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    data["chain_Ree"].append(Ree)
    data["chain_Rg"].append(Rg)
    data["chain_bend"].append(bend)
    data["chain_ring_tilt"].append(tilt)

    data["ring_breath"].append(ring_breath)
    data["ring_asym"].append(ring_asym)
    data["ring_pucker"].append(ring_pucker)

    data["C13_N4_bond"].append(bond_CN)
    data["C14_C13_N4_angle"].append(ang_attach)

    data["attach_sin"].append(
        np.sin(np.radians(tors_attach))
    )

    data["attach_cos"].append(
        np.cos(np.radians(tors_attach))
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(data)

tempo_cols = [

    "NO_bond",
    "ONO_angle",

    "tors1_sin",
    "tors1_cos",

    "tors2_sin",
    "tors2_cos"
]

other_cols = [

    c for c in df.columns
    if c not in tempo_cols
]


# ============================================================
# MUTUAL INFORMATION
# ============================================================

MI = np.zeros(
    (
        len(other_cols),
        len(tempo_cols)
    )
)

for i, xcol in enumerate(other_cols):

    X = df[xcol].values.reshape(-1,1)

    for j, ycol in enumerate(tempo_cols):

        y = df[ycol].values

        MI[i,j] = mutual_info_regression(
            X,
            y,
            n_neighbors=10,
            random_state=0
        )[0]


MI_df = pd.DataFrame(
    MI,
    index=other_cols,
    columns=tempo_cols
)

print("\n")
print("="*60)
print("MUTUAL INFORMATION MATRIX")
print("="*60)
print(MI_df.round(4))


# ============================================================
# SAVE CSV
# ============================================================

MI_df.to_csv(
    "TEMPO_MI_matrix.csv"
)


# ============================================================
# HEATMAP
# ============================================================

plt.figure(
    figsize=(10,8)
)

im = plt.imshow(
    MI,
    aspect="auto"
)

plt.colorbar(
    label="Mutual Information"
)

plt.xticks(
    np.arange(len(tempo_cols)),
    tempo_cols,
    rotation=45,
    ha="right",
    fontsize=24,
    fontweight="bold"
)

plt.yticks(
    np.arange(len(other_cols)),
    other_cols,
    fontsize=24,
    fontweight="bold"
)


for i in range(MI.shape[0]):
    for j in range(MI.shape[1]):

        plt.text(
            j,
            i,
            f"{MI[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    "TEMPO_MI_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()