"""
Paper 1 — Extended Baselines (B5, B6, B7, B8, B9)
===================================================
Implements the advanced baselines for comparison against beta-interpolation.
"""

import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pennylane import numpy as np
import pennylane as qml
import pickle
import os
from tqdm import tqdm

# Import shared config and exact state
from step4_vqe_optimization import (
    N_QUBITS, L_LAYERS, H_FIELD, H_FIELD_Z, STEPS, LR, SEEDS, 
    exact_state, exact_energy, H_pl
)

# --- Global Device ---
dev = qml.device("default.qubit", wires=N_QUBITS)

def save_checkpoint(results):
    os.makedirs('results', exist_ok=True)
    with open('results/baselines_extended_results.pkl', 'wb') as f:
        pickle.dump(results, f)

# --- B5: Horizontal Gates (Geometric Approach) ---
def count_params_b5(n, L):
    return L * ((n - 1) + n + n) # ZZ + X (sym) + Y (horizontal)

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_b5_energy(theta):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    idx = 0
    for _ in range(L_LAYERS):
        # Symmetric operations
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx], wires=i); idx += 1
        # Horizontal (symmetry-breaking) perturbations
        for i in range(N_QUBITS):
            qml.RY(2.0 * theta[idx], wires=i); idx += 1
    return qml.expval(H_pl)

@qml.qnode(dev, interface="numpy")
def circuit_b5_state(theta):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1]); idx += 1
        for i in range(N_QUBITS):
            qml.RX(2.0 * theta[idx], wires=i); idx += 1
        for i in range(N_QUBITS):
            qml.RY(2.0 * theta[idx], wires=i); idx += 1
    return qml.state()

def run_b5(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params_b5(N_QUBITS, L_LAYERS)), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(circuit_b5_energy, theta)
        if step == STEPS - 1:
            state = circuit_b5_state(theta)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid)))
    return history[-1]

# --- B6: Standard HEA (RY, RZ, CNOT) ---
def count_params_b6(n, L):
    return L * 2 * n

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_b6_energy(theta):
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(theta[idx], wires=i); idx += 1
            qml.RZ(theta[idx], wires=i); idx += 1
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(H_pl)

@qml.qnode(dev, interface="numpy")
def circuit_b6_state(theta):
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(theta[idx], wires=i); idx += 1
            qml.RZ(theta[idx], wires=i); idx += 1
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.state()

def run_b6(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params_b6(N_QUBITS, L_LAYERS)), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(circuit_b6_energy, theta)
        if step == STEPS - 1:
            state = circuit_b6_state(theta)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid)))
    return history[-1]

# --- B7: QAOA-Style ---
def count_params_b7(L):
    return 2 * L

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_b7_energy(theta):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    for l in range(L_LAYERS):
        gamma = theta[2*l]
        beta = theta[2*l+1]
        # Target Hamiltonian evolution (gamma)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i+1]); qml.RZ(2.0 * gamma, wires=i+1); qml.CNOT(wires=[i, i+1])
        for i in range(N_QUBITS):
            qml.RZ(2.0 * H_FIELD_Z * gamma, wires=i)
        for i in range(N_QUBITS):
            qml.RX(2.0 * H_FIELD * gamma, wires=i)
        # Mixer evolution (beta)
        for i in range(N_QUBITS):
            qml.RY(2.0 * beta, wires=i)
    return qml.expval(H_pl)

@qml.qnode(dev, interface="numpy")
def circuit_b7_state(theta):
    for i in range(N_QUBITS): qml.Hadamard(wires=i)
    for l in range(L_LAYERS):
        gamma = theta[2*l]; beta = theta[2*l+1]
        for i in range(N_QUBITS - 1): qml.CNOT(wires=[i, i+1]); qml.RZ(2.0 * gamma, wires=i+1); qml.CNOT(wires=[i, i+1])
        for i in range(N_QUBITS): qml.RZ(2.0 * H_FIELD_Z * gamma, wires=i)
        for i in range(N_QUBITS): qml.RX(2.0 * H_FIELD * gamma, wires=i)
        for i in range(N_QUBITS): qml.RY(2.0 * beta, wires=i)
    return qml.state()

def run_b7(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params_b7(L_LAYERS)), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(circuit_b7_energy, theta)
        if step == STEPS - 1:
            state = circuit_b7_state(theta)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid)))
    return history[-1]

# --- B8 & B9: ADAPT-VQE (Simplified Pool) ---
def run_adapt(seed, use_delta=False):
    """
    Simplified ADAPT-VQE. 
    Pool: RY on each qubit, ZZ on nearest neighbors.
    Max iterations: L_LAYERS.
    """
    pool_ops = [('RY', i) for i in range(N_QUBITS)] + \
               [('ZZ', i) for i in range(N_QUBITS - 1)]
               
    np.random.seed(seed)
    selected_ops = []
    theta = np.array([], requires_grad=True)
    
    def adapt_circuit(params):
        for i in range(N_QUBITS): qml.Hadamard(wires=i)
        for idx, op in enumerate(selected_ops):
            if op[0] == 'RY': qml.RY(params[idx], wires=op[1])
            elif op[0] == 'ZZ': 
                qml.CNOT(wires=[op[1], op[1]+1])
                qml.RZ(params[idx], wires=op[1]+1)
                qml.CNOT(wires=[op[1], op[1]+1])
            
    @qml.qnode(dev, interface="numpy")
    def adapt_energy(params):
        adapt_circuit(params)
        return qml.expval(H_pl)
        
    @qml.qnode(dev, interface="numpy")
    def adapt_state(params):
        adapt_circuit(params)
        return qml.state()
        
    for l in range(L_LAYERS):
        best_grad = -1; best_op = None
        
        # Measure gradient for all pool ops
        for op in pool_ops:
            temp_ops = selected_ops + [op]
            temp_theta = np.append(theta, 0.0)
            temp_theta.requires_grad = True
            
            @qml.qnode(dev, interface="numpy")
            def temp_energy(p):
                for i in range(N_QUBITS): qml.Hadamard(wires=i)
                for idx, o in enumerate(temp_ops):
                    if o[0] == 'RY': qml.RY(p[idx], wires=o[1])
                    elif o[0] == 'ZZ': 
                        qml.CNOT(wires=[o[1], o[1]+1])
                        qml.RZ(p[idx], wires=o[1]+1)
                        qml.CNOT(wires=[o[1], o[1]+1])
                return qml.expval(H_pl)
                
            grad = np.abs(qml.grad(temp_energy)(temp_theta)[-1])
            if use_delta:
                # Delta weighting heuristic: weight by degree of symmetry breaking
                weight = 1.0 if op[0] == 'RY' else 0.5 
                grad *= weight
                
            if grad > best_grad:
                best_grad = grad
                best_op = op
                
        if best_op is None or best_grad < 1e-4: break
        
        selected_ops.append(best_op)
        theta = np.append(theta, 0.0)
        theta.requires_grad = True
        
        # Optimize circuit with new op
        opt = qml.AdamOptimizer(LR)
        for _ in range(30): # Short optimization per macro-step
            theta, _ = opt.step_and_cost(adapt_energy, theta)
            
    # Final full optimization
    opt = qml.AdamOptimizer(LR)
    for _ in range(STEPS):
        theta, e = opt.step_and_cost(adapt_energy, theta)
        
    state = adapt_state(theta)
    fid = np.abs(np.vdot(exact_state, state))**2
    return (float(e), float(fid))

if __name__ == '__main__':
    res_path = 'results/baselines_extended_results.pkl'
    if os.path.exists(res_path):
        with open(res_path, 'rb') as f: results = pickle.load(f)
    else:
        results = {s: {} for s in SEEDS}
        
    for seed in SEEDS:
        print(f"\n--- Running Extended Baselines Seed {seed} ---")
        
        # Initialize seed dict if missing (e.g. expanding from 3 to 7 seeds)
        if seed not in results:
            results[seed] = {}
        
        if 'B5' not in results[seed]:
            print("Running B5 (Horizontal)..."); results[seed]['B5'] = run_b5(seed); save_checkpoint(results)
            
        if 'B6' not in results[seed]:
            print("Running B6 (Standard HEA)..."); results[seed]['B6'] = run_b6(seed); save_checkpoint(results)
            
        if 'B7' not in results[seed]:
            print("Running B7 (QAOA)..."); results[seed]['B7'] = run_b7(seed); save_checkpoint(results)
            
        if 'B8' not in results[seed]:
            print("Running B8 (ADAPT-VQE)..."); results[seed]['B8'] = run_adapt(seed, False); save_checkpoint(results)
            
        if 'B9' not in results[seed]:
            print("Running B9 (ADAPT-VQE with Delta)..."); results[seed]['B9'] = run_adapt(seed, True); save_checkpoint(results)
            
    print("\n✅ Extended Baselines Complete.")
