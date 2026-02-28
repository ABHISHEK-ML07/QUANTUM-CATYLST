# src/vqe.py
"""
VQE-style script (no quantum backend required).
- Mode 'exact' computes exact ground state via numpy.linalg.eigh
- Mode 'vqe' runs a classical variational optimization (product-state ansatz)
Outputs saved to results/
"""
import os
import json
import numpy as np
from time import time

os.makedirs("results", exist_ok=True)
np.random.seed(12345)

H_PATH = "results/hamiltonian.npy"

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
    """
    Build product-state vector from angles (one angle per qubit).
    Single-qubit state = [cos(theta/2), sin(theta/2)]
    Full state = Kron over qubits.
    angles: array-like length n_qubits
    returns normalized complex vector of length 2^n
    """
    states = []
    for th in angles:
        a = np.cos(th/2.0)
        b = np.sin(th/2.0)
        states.append(np.array([a, b], dtype=complex))
    # Kronecker product
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    # normalize (should be normalized already)
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
    except Exception as e:
        print("scipy not available, falling back to random search.")
        return vqe_random_search(H, n_qubits)
    def obj(x):
        return expectation_from_angles(H, x)
    x0 = np.zeros(n_qubits)
    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter":1000, "disp": False})
    val = float(res.fun)
    return val, res.x, res.nit if hasattr(res, "nit") else 0

def main():
    H = load_h()
    n_states = H.shape[0]
    n_qubits = int(np.log2(n_states))
    print(f"Loaded Hamiltonian: matrix shape {H.shape}, inferred qubits: {n_qubits}")

    # Exact diagonalization
    print("Running exact diagonalization...")
    eg, evec = exact_diagonalization(H)
    print(f"Exact ground energy: {eg:.8f}")
    # save exact
    np.save("results/exact_ground.npy", np.array([eg]))
    np.save("results/exact_state.npy", evec)
    with open("results/exact_info.json", "w") as f:
        json.dump({"exact_ground": float(eg), "n_qubits": n_qubits}, f, indent=2)

    # VQE (classical product-state)
    print("Running VQE (product-state ansatz)...")
    # prefer scipy if available
    try:
        import scipy
        use_scipy = True
    except:
        use_scipy = False

    if use_scipy:
        print("scipy detected: using Nelder-Mead optimizer (scipy.optimize.minimize).")
        val, angles, extra = vqe_with_scipy(H, n_qubits)
        method = "scipy_nelder_mead"
    else:
        print("scipy not detected: using random-search optimizer (pure numpy).")
        val, angles, runtime = vqe_random_search(H, n_qubits, n_iters=3000, step_scale=0.6)
        method = "random_search"
    print(f"VQE (ansatz) energy: {val:.8f}  (method: {method})")
    # Save VQE results
    np.save("results/vqe_energy.npy", np.array([val]))
    np.save("results/vqe_angles.npy", np.array(angles))
    with open("results/vqe_info.json", "w") as f:
        json.dump({"vqe_energy": float(val), "method": method, "n_qubits": n_qubits}, f, indent=2)

    # Compare
    gap = val - eg
    print(f"VQE - Exact gap: {gap:.8f}")

    # Save a small leaderboard-like CSV
    import csv
    with open("results/leaderboard.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material", "method", "energy_eV"])
        writer.writerow(["graphene_N_doped (cluster)", "exact", f"{eg:.8f}"])
        writer.writerow(["graphene_N_doped (cluster)", "vqe_ansatz", f"{val:.8f}"])

    print("Saved results to results/ (exact_ground.npy, vqe_energy.npy, leaderboard.csv, ...)")
    print("Done.")

if __name__ == "__main__":
    main()