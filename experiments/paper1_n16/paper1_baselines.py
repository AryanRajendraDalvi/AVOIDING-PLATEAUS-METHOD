"""
Paper 1 — Baselines and Ablations
===================================
Runs the baseline comparisons and ablation studies.

B3: Fixed symmetry-breaking layer (Park 2021).
A1: Per-layer beta parameters (instead of a global scalar beta).
"""

import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pennylane import numpy as np
import pennylane as qml
import pickle
import os
from tqdm import tqdm

from step4_vqe_optimization import get_exact_ground_state, H_pl, N_QUBITS, L_LAYERS, H_FIELD, STEPS, LR, count_params

SEEDS = [42, 43, 44]
exact_energy, exact_state = get_exact_ground_state(N_QUBITS, H_FIELD)
# --- Global Device ---
dev = qml.device("default.qubit", wires=N_QUBITS)

# --- B3: Fixed Symmetry-Breaking Layer ---
@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_b3_energy(theta, theta_break):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
        
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx], wires=i) # purely symmetric
            idx += 1
            
    # Fixed breaking layer at the end
    for i in range(N_QUBITS):
        qml.RY(2.0 * theta_break[i], wires=i)
        
    return qml.expval(H_pl)

@qml.qnode(dev, interface="numpy")
def circuit_b3_state(theta, theta_break):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx], wires=i)
            idx += 1
    for i in range(N_QUBITS):
        qml.RY(2.0 * theta_break[i], wires=i)
    return qml.state()

def run_baseline_b3(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    theta_break = np.array(np.random.uniform(-np.pi, np.pi, N_QUBITS), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    
    def cost(t, tb): return circuit_b3_energy(t, tb)
    
    history = []
    for step in range(STEPS):
        (theta, theta_break), e = opt.step_and_cost(cost, theta, theta_break)
        if step == STEPS - 1:
            state = circuit_b3_state(theta, theta_break)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid)))
    return history[-1]

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_a1_energy(theta, betas):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
        
    idx = 0
    for l in range(L_LAYERS):
        beta = betas[l]
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx] * (1.0 - beta), wires=i)
            qml.RY(2.0 * theta[idx] * beta, wires=i)
            idx += 1
    return qml.expval(H_pl)

@qml.qnode(dev, interface="numpy")
def circuit_a1_state(theta, betas):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    idx = 0
    for l in range(L_LAYERS):
        beta = betas[l]
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx] * (1.0 - beta), wires=i)
            qml.RY(2.0 * theta[idx] * beta, wires=i)
            idx += 1
    return qml.state()

def run_ablation_a1(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    betas = np.array(np.full(L_LAYERS, 0.5), requires_grad=True) # start middle
    opt = qml.AdamOptimizer(LR)
    
    def cost(t, b): return circuit_a1_energy(t, b)
    
    for step in range(STEPS):
        # Update theta and betas jointly via ultra-fast Adjoint AD
        g_theta, g_betas = qml.grad(circuit_a1_energy, argnums=(0, 1))(theta, betas)
        theta = theta - LR * g_theta
        betas = np.clip(betas - LR * g_betas, 0.0, 1.0)
        
    state = circuit_a1_state(theta, betas)
    fid = np.abs(np.vdot(exact_state, state))**2
    e = circuit_a1_energy(theta, betas)
    return (float(e), float(fid), betas.tolist())

if __name__ == '__main__':
    res_path = 'results/baselines_results.pkl'
    if os.path.exists(res_path):
        print(f"Loading checkpoint from {res_path}... Baselines B3 and A1 already complete. Skipping.")
        with open(res_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {'B3': [], 'A1': []}
        for seed in SEEDS:
            print(f"Running Baselines for Seed {seed}...")
            results['B3'].append(run_baseline_b3(seed))
            results['A1'].append(run_ablation_a1(seed))
            
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        
    print("\n✅ Baselines Complete.")
    
    # Print Summary
    b3_fids = [r[1] for r in results['B3']]
    a1_fids = [r[1] for r in results['A1']]
    print(f"Baseline B3 (Fixed Layer) Mean Fidelity: {np.mean(b3_fids):.3f}")
    print(f"Ablation A1 (Per-layer Beta) Mean Fidelity: {np.mean(a1_fids):.3f}")
