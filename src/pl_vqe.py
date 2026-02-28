# src/pl_vqe.py
"""
PennyLane VQE that reads a full Hamiltonian matrix from results/hamiltonian.npy
and uses qml.Hermitian to evaluate <psi|H|psi> for a parametrized ansatz.
Saves results to results/pl_vqe_info.json and results/pl_vqe_energy.npy
"""
import json, pathlib, sys
project_root = str(pathlib.Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pathlib import Path
from src.vqe import product_state_from_angles

H_PATH = Path("results/hamiltonian.npy")
OUT_JSON = Path("results/pl_vqe_info.json")
OUT_NPY = Path("results/pl_vqe_energy.npy")

def load_h():
    if not H_PATH.exists():
        raise FileNotFoundError("Hamiltonian not found at results/hamiltonian.npy")
    H = np.load(str(H_PATH))
    return H

def ansatz(params, wires):
    # simple rotated product-state ansatz implemented as per-angle rotations
    for i, th in enumerate(params):
        qml.RY(th, wires=i)
    # lightweight entanglement
    for i in range(len(wires)-1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def run_pennylane_vqe(maxiter=150, lr=0.15):
    H = load_h()
    n_states = H.shape[0]
    n_qubits = int(np.log2(n_states))
    dev = qml.device("default.qubit", wires=n_qubits)
    # Convert H to a Hermitian observable for Pennylane
    H_mat = H.astype(np.complex128)

    @qml.qnode(dev, interface="autograd")
    def circuit(params):
        ansatz(params, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(H_mat, wires=range(n_qubits)))

    # initialize params
    params = pnp.zeros(n_qubits, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    energy = None
    for it in range(maxiter):
        params, energy = opt.step_and_cost(circuit, params)
        if it % 25 == 0:
            print(f"Iter {it:03d}: energy = {energy:.8f}")
    # final
    print("Final energy (PennyLane VQE):", float(energy))
    # save
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT_NPY), np.array([float(energy)]))
    OUT_JSON.write_text(json.dumps({"pl_vqe_energy": float(energy), "n_qubits": n_qubits}, indent=2))
    return float(energy)

if __name__ == "__main__":
    e = run_pennylane_vqe(maxiter=120)
    print("Done. Energy:", e)