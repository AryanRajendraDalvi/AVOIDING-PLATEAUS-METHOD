"""
Paper 1 — Ablations (A3, A5, A6)
==================================
Implements A3 (Generator Choice), A5 (Shot Noise), and A6 (U(1) Symmetry).
(Note: A1 is in paper1_baselines.py)
"""

import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pennylane import numpy as np
import pennylane as qml
import pickle
import os
from tqdm import tqdm

from step4_vqe_optimization import (
    N_QUBITS, L_LAYERS, H_FIELD, H_FIELD_Z, STEPS, LR, SEEDS, count_params, H_pl
)

dev = qml.device("default.qubit", wires=N_QUBITS)

def save_checkpoint(results):
    os.makedirs('results', exist_ok=True)
    with open('results/ablations_results.pkl', 'wb') as f:
        pickle.dump(results, f)

# --- A2: Optimizer Choice (Vanilla SGD) ---
def run_a2(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    beta = 0.5
    
    @qml.qnode(dev, interface="numpy", diff_method="adjoint")
    def circ_a2(t):
        for i in range(N_QUBITS): qml.Hadamard(wires=i)
        idx = 0
        for _ in range(L_LAYERS):
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * t[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
            for i in range(N_QUBITS):
                qml.RX(2.0 * t[idx] * (1.0 - beta), wires=i)
                qml.RZ(2.0 * t[idx] * beta, wires=i)
                idx += 1
        return qml.expval(H_pl)
        
    opt = qml.GradientDescentOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(circ_a2, theta)
        if step == STEPS - 1: history.append(float(e))
    return history[-1]

# --- A3: Generator Choice (Z vs Y vs ZZ) ---
def run_a3_generator(seed, gen_type):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    beta = 0.5
    
    @qml.qnode(dev, interface="numpy", diff_method="adjoint")
    def circ(t):
        for i in range(N_QUBITS): qml.Hadamard(wires=i)
        idx = 0
        for _ in range(L_LAYERS):
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * t[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
            for i in range(N_QUBITS):
                angle_x = 2.0 * t[idx] * (1.0 - beta)
                angle_brk = 2.0 * t[idx] * beta
                qml.RX(angle_x, wires=i)
                if gen_type == 'Y': qml.RY(angle_brk, wires=i)
                elif gen_type == 'Z': qml.RZ(angle_brk, wires=i)
                elif gen_type == 'ZZ':
                    if i < N_QUBITS - 1: qml.CNOT(wires=[i, i+1]); qml.RZ(angle_brk, wires=i+1); qml.CNOT(wires=[i, i+1])
                idx += 1
        return qml.expval(H_pl)
        
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(circ, theta)
        if step == STEPS - 1: history.append(float(e))
    return history[-1]

# --- A4: Schedule Shape (Exponential Ramp) ---
def run_a4(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    
    @qml.qnode(dev, interface="numpy", diff_method="adjoint")
    def circ_a4(t, beta_val):
        for i in range(N_QUBITS): qml.Hadamard(wires=i)
        idx = 0
        for _ in range(L_LAYERS):
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * t[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
            for i in range(N_QUBITS):
                qml.RX(2.0 * t[idx] * (1.0 - beta_val), wires=i)
                qml.RZ(2.0 * t[idx] * beta_val, wires=i)
                idx += 1
        return qml.expval(H_pl)
        
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        # Exponential schedule: (exp(3*step/STEPS)-1)/(exp(3)-1)
        b = (np.exp(3.0 * step / STEPS) - 1.0) / (np.exp(3.0) - 1.0)
        theta, e = opt.step_and_cost(lambda t: circ_a4(t, b), theta)
        if step == STEPS - 1: history.append(float(e))
    return history[-1]

# --- A5: Shot Noise (Simulated via Noisy Gradients to bypass Autograd bug) ---
@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_a5_analytic(theta, beta):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1): qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx] * (1.0 - beta), wires=i)
            qml.RY(2.0 * theta[idx] * beta, wires=i)
            idx += 1
    return qml.expval(H_pl)

def run_a5(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    beta = 0.5
    
    # Mathematical derivation of parameter-shift gradient noise:
    # Var(g) = Var(E(+) - E(-)) / 4 = (Var(E+) + Var(E-)) / 4 = 2 * Var(E) / 4 = Var(E) / 2
    shots = 1000
    noise_std = np.sqrt(len(H_pl.coeffs)) / np.sqrt(shots)
    grad_noise_std = noise_std / np.sqrt(2)
    
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    
    for step in range(1, STEPS + 1):
        g_exact = qml.grad(circuit_a5_analytic, argnums=0)(theta, beta)
        g_noisy = g_exact + np.random.normal(0, grad_noise_std, size=len(theta))
        
        m = b1 * m + (1 - b1) * g_noisy
        v = b2 * v + (1 - b2) * (g_noisy ** 2)
        m_hat = m / (1 - b1**step)
        v_hat = v / (1 - b2**step)
        
        theta = theta - LR * m_hat / (np.sqrt(v_hat) + eps)
        
    e_final = circuit_a5_analytic(theta, beta)
    return float(e_final)

# --- A6: U(1) Symmetry (XXZ Model) ---
# H = - \sum (XX + YY) + Delta \sum ZZ
H_u1_coeffs = [-1.0] * (N_QUBITS - 1) * 2 + [0.5] * (N_QUBITS - 1)
H_u1_obs = [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(N_QUBITS - 1)] + \
           [qml.PauliY(i) @ qml.PauliY(i+1) for i in range(N_QUBITS - 1)] + \
           [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(N_QUBITS - 1)]
H_u1 = qml.Hamiltonian(H_u1_coeffs, H_u1_obs)

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_a6_u1(theta, beta):
    for i in range(N_QUBITS):
        if i % 2 == 0: qml.PauliX(wires=i) # Neel state initialization (Total Z=0)
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.IsingXX(2.0 * theta[idx], wires=[i, i+1])
            qml.IsingYY(2.0 * theta[idx], wires=[i, i+1]); idx += 1
        for i in range(N_QUBITS):
            qml.RZ(2.0 * theta[idx] * (1.0 - beta), wires=i) # Symmetric
            qml.RX(2.0 * theta[idx] * beta, wires=i)         # Breaking (does not commute with total Z)
            idx += 1
    return qml.expval(H_u1)

def run_a6(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, L_LAYERS*(N_QUBITS-1 + N_QUBITS)), requires_grad=True)
    beta = 0.5
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(lambda t: circuit_a6_u1(t, beta), theta)
        if step == STEPS - 1: history.append(float(e))
    return history[-1]

if __name__ == '__main__':
    res_path = 'results/ablations_results.pkl'
    if os.path.exists(res_path):
        with open(res_path, 'rb') as f: results = pickle.load(f)
    else:
        results = {s: {} for s in SEEDS}
        
    for seed in SEEDS:
        print(f"\n--- Running Ablations Seed {seed} ---")
        
        # Initialize seed dict if missing (e.g. expanding from 3 to 7 seeds)
        if seed not in results:
            results[seed] = {}
        
        if 'A2' not in results[seed]:
            print("Running A2 (SGD Optimizer)..."); results[seed]['A2'] = run_a2(seed); save_checkpoint(results)
            
        if 'A3_Z' not in results[seed]:
            print("Running A3 (Gen: Z)..."); results[seed]['A3_Z'] = run_a3_generator(seed, 'Z'); save_checkpoint(results)
        if 'A3_Y' not in results[seed]:
            print("Running A3 (Gen: Y)..."); results[seed]['A3_Y'] = run_a3_generator(seed, 'Y'); save_checkpoint(results)
        if 'A3_ZZ' not in results[seed]:
            print("Running A3 (Gen: ZZ)..."); results[seed]['A3_ZZ'] = run_a3_generator(seed, 'ZZ'); save_checkpoint(results)
            
        if 'A4' not in results[seed]:
            print("Running A4 (Exp Schedule)..."); results[seed]['A4'] = run_a4(seed); save_checkpoint(results)
            
        if 'A5' not in results[seed]:
            print("Running A5 (Shot Noise)..."); results[seed]['A5'] = run_a5(seed); save_checkpoint(results)
            
        if 'A6' not in results[seed]:
            print("Running A6 (U(1) Hubbard)..."); results[seed]['A6'] = run_a6(seed); save_checkpoint(results)
            
    print("\n✅ Ablations Complete.")
