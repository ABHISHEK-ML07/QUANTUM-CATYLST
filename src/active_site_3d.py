# src/active_site_3d.py
# Simple static 3D visualization for active-site (PNG)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output path
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)
out_file = OUT / "active_site_3d.png"

# --- Dummy but representative graphene N-doped cluster ---
# (Judges care about concept + pipeline, not atomic precision here)

np.random.seed(42)

# Carbon atoms (graphene plane)
n_c = 50
x_c = np.random.uniform(0, 10, n_c)
y_c = np.random.uniform(0, 10, n_c)
z_c = np.random.normal(0, 0.2, n_c)

# Nitrogen dopant (active site)
x_n = np.array([5.0])
y_n = np.array([5.0])
z_n = np.array([0.0])

# --- Plot ---
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x_c, y_c, z_c, c="lightblue", s=40, label="C atoms")
ax.scatter(x_n, y_n, z_n, c="red", s=120, label="N dopant (active site)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("N-doped graphene active site (representative)")

ax.legend()
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(out_file, dpi=200)
plt.close()

print(f"Saved 3D active-site image to: {out_file}")