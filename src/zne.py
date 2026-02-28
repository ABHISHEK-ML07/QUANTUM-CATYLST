# src/zne.py
"""
Noise emulation + Zero-Noise Extrapolation (ZNE) script.

- Uses results/hamiltonian.npy created earlier.
- Emulates noise by adding a small, reproducible Hermitian random matrix scaled by (scale-1)*noise_strength.
- Runs exact diagonalization and product-state VQE (same style as src/vqe.py) for each noise scale.
- Performs Richardson (linear) extrapolation to estimate zero-noise energy.
- Saves JSON + CSV outputs to results/.
"""
import os
import json
import numpy as np
from time import time
np.random.seed(12345)

H_PATH = "results/hamiltonian.npy"
OUT_JSON = "results/zne_results.json"
OUT_CSV = "results/zne_leaderboard.csv"

# Noise / scales
SCALES = [1.0, 2.0, 3.0]           # noise multipliers
NOISE_REL_STRENGTH = 0.01          # relative to norm(H); tuneable

# --- Utility functions (duplicated from vqe.py style for standalone run) ---
def load_h():
    if not os.path.exists(H_PATH):
        raise FileNotFoundError(f"Hamiltonian not found at {H_PATH}")
    H = np.load(H_PATH)
    return H

def exact_diagonalization(H):
    vals, vecs = np.linalg.eigh(H)
    idx = np.argmin(vals)
    return float(vals[idx]), vecs[:, idx]

def product_state_from_angles(angles):
    states = []
    for th in angles:
        a = np.cos(th/2.0)
        b = np.sin(th/2.0)
        states.append(np.array([a, b], dtype=complex))
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    psi = psi / np.linalg.norm(psi)
    return psi

def expectation_from_angles(H, angles):
    psi = product_state_from_angles(angles)
    e = np.vdot(psi, H.dot(psi)).real
    return float(e)

def vqe_random_search(H, n_qubits, n_iters=2000, step_scale=0.5):
    best_angles = np.zeros(n_qubits)
    best_val = expectation_from_angles(H, best_angles)
    t0 = time()
    for i in range(n_iters):
        cand = best_angles + np.random.normal(scale=step_scale, size=n_qubits)
        val = expectation_from_angles(H, cand)
        if val < best_val:
            best_val = val
            best_angles = cand
    return best_val, best_angles, time() - t0

def vqe_with_scipy(H, n_qubits):
    try:
        from scipy.optimize import minimize
    except Exception:
        return vqe_random_search(H, n_qubits)
    def obj(x):
        return expectation_from_angles(H, x)
    x0 = np.zeros(n_qubits)
    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter":1500, "disp": False})
    val = float(res.fun)
    return val, res.x, res.nit if hasattr(res, "nit") else 0

# --- Noise builder ---
def make_noise_matrix(n, seed=999):
    """Create a random Hermitian matrix with unit Frobenius norm (deterministic via seed)."""
    rs = np.random.RandomState(seed)
    A = rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n))
    H_rand = (A + A.conj().T) / 2.0
    # normalize
    fro = np.linalg.norm(H_rand, ord='fro')
    if fro == 0:
        return H_rand
    return H_rand / fro

def add_noise(H, scale, noise_rel_strength=NOISE_REL_STRENGTH):
    """Return noisy Hamiltonian H_noisy = H + (scale-1)*noise_strength*R where R is unit-norm Hermitian."""
    normH = np.linalg.norm(H, ord='fro')
    R = make_noise_matrix(H.shape[0], seed=42)  # deterministic
    noise_amp = noise_rel_strength * normH * (scale - 1.0)
    H_noisy = H + noise_amp * R
    return H_noisy

# --- Simple Richardson (linear) extrapolation ---
def richardson_extrapolate(scales, values):
    """
    Fit a linear polynomial (degree 1) to values vs scales and evaluate at 0.
    Returns extrapolated_value and fit coefficients.
    """
    coefs = np.polyfit(scales, values, 1)  # degree 1
    extrap = np.polyval(coefs, 0.0)
    return float(extrap), coefs.tolist()

def main():
    os.makedirs("results", exist_ok=True)
    H = load_h()
    n_states = H.shape[0]
    n_qubits = int(np.log2(n_states))
    print(f"Loaded H shape {H.shape}, n_qubits {n_qubits}")

    results = {"scales": [], "exact": [], "vqe": []}

    for s in SCALES:
        print(f"\n--- Running scale {s} ---")
        Hs = add_noise(H, s, noise_rel_strength=NOISE_REL_STRENGTH)
        eg, evec = exact_diagonalization(Hs)
        # run VQE-style variational (prefers scipy but falls back)
        try:
            import scipy
            use_scipy = True
        except:
            use_scipy = False

        if use_scipy:
            val, angles, extra = vqe_with_scipy(Hs, n_qubits)
            method = "scipy_nelder_mead"
        else:
            val, angles, runtime = vqe_random_search(Hs, n_qubits, n_iters=2500, step_scale=0.6)
            method = "random_search"

        print(f"Scale {s}: exact={eg:.8f}, vqe={val:.8f} (method={method})")
        results["scales"].append(s)
        results["exact"].append(float(eg))
        results["vqe"].append(float(val))

    # Extrapolate to zero-noise
    exact_extrap, exact_coefs = richardson_extrapolate(results["scales"], results["exact"])
    vqe_extrap, vqe_coefs = richardson_extrapolate(results["scales"], results["vqe"])

    summary = {
        "scales": results["scales"],
        "exact_values": results["exact"],
        "vqe_values": results["vqe"],
        "exact_extrapolated": exact_extrap,
        "exact_fit_coefs": exact_coefs,
        "vqe_extrapolated": vqe_extrap,
        "vqe_fit_coefs": vqe_coefs,
        "noise_rel_strength": NOISE_REL_STRENGTH
    }

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV leaderboard (for one material)
    import csv
    with open(OUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material", "method", "energy"])
        writer.writerow(["graphene_N_doped_cluster", "exact_scale1", f"{results['exact'][0]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "exact_scale2", f"{results['exact'][1]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "exact_scale3", f"{results['exact'][2]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "exact_extrapolated", f"{summary['exact_extrapolated']:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "vqe_scale1", f"{results['vqe'][0]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "vqe_scale2", f"{results['vqe'][1]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "vqe_scale3", f"{results['vqe'][2]:.8f}"])
        writer.writerow(["graphene_N_doped_cluster", "vqe_extrapolated", f"{summary['vqe_extrapolated']:.8f}"])

    print("\nZNE complete. JSON and CSV saved to results/")
    print(json.dumps(summary, indent=2))
    print("Done.")

if __name__ == '__main__':
    main()