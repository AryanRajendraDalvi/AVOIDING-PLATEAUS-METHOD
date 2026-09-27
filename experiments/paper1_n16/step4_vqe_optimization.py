"""
Paper 1 — Step 4: Full VQE Optimization
=========================================
Runs VQE optimization for the Transverse-Field Ising Model (TFIM)
across different beta strategies to test Hypothesis 4.16.

Strategies:
  - S0: Fixed beta = 0.0 (Strictly Equivariant)
  - S1: Fixed beta in (0, 1) (Interpolated)
  - S1_1: Fixed beta = 1.0 (Fully Unrestricted)
  - S2: Scheduled beta (ramp from 0 to 1)
  - S3: Learned beta (regularized by Commutator-Defect Delta)
"""

import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pennylane import numpy as np
import pennylane as qml
import pickle
import os
from tqdm import tqdm

# --- Configuration ---
N_QUBITS = 16
L_LAYERS = 2 * N_QUBITS
H_FIELD = 0.5
H_FIELD_Z = 0.1  # The symmetry-breaking longitudinal field fix
STEPS = 500
LR = 0.01
SEEDS = [42, 43, 44]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

np.set_printoptions(precision=4, suppress=True)

# --- 1. Exact Diagonalization (TFIM) ---
def get_tfim_hamiltonian_matrix(n, h):
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    H = np.zeros((2**n, 2**n), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.eye(1, dtype=complex)
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H -= term
        
    # X terms
    for i in range(n):
        term = np.eye(1, dtype=complex)
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H -= h * term
        
    # Z terms (Longitudinal field)
    for i in range(n):
        term = np.eye(1, dtype=complex)
        for j in range(n):
            if j == i:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H -= H_FIELD_Z * term
        
    return H

def get_exact_ground_state(n, h):
    H = get_tfim_hamiltonian_matrix(n, h)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals[0], eigvecs[:, 0]

exact_energy, exact_state = get_exact_ground_state(N_QUBITS, H_FIELD)
print(f"Exact Ground State Energy: {exact_energy:.6f}")

# --- 2. VQE Circuit ---
dev = qml.device("default.qubit", wires=N_QUBITS)

def count_params(n, L):
    return L * ((n - 1) + n)

@qml.qnode(dev, interface="numpy")
def circuit_state(theta, beta):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
            
        for i in range(N_QUBITS):
            angle_x = 2.0 * theta[idx] * (1.0 - beta)
            angle_y = 2.0 * theta[idx] * beta
            if abs(angle_x) > 1e-10:
                qml.RX(angle_x, wires=i)
            if abs(angle_y) > 1e-10:
                qml.RY(angle_y, wires=i) # Using Y for breaking (as established in Step 1)
            idx += 1
            
    return qml.state()

# Hamiltonians for Pennylane
H_coeffs = [-1.0] * (N_QUBITS - 1) + [-H_FIELD] * N_QUBITS + [-H_FIELD_Z] * N_QUBITS
H_obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(N_QUBITS - 1)] + \
        [qml.PauliX(i) for i in range(N_QUBITS)] + \
        [qml.PauliZ(i) for i in range(N_QUBITS)]
H_pl = qml.Hamiltonian(H_coeffs, H_obs)

@qml.qnode(dev, interface="numpy", diff_method="adjoint")
def circuit_energy(theta, beta):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        
    idx = 0
    for _ in range(L_LAYERS):
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1]); qml.RZ(2.0 * theta[idx], wires=i + 1); qml.CNOT(wires=[i, i + 1])
            idx += 1
            
        for i in range(N_QUBITS):
            angle_x = 2.0 * theta[idx] * (1.0 - beta)
            angle_y = 2.0 * theta[idx] * beta
            if abs(angle_x) > 1e-10:
                qml.RX(angle_x, wires=i)
            if abs(angle_y) > 1e-10:
                qml.RY(angle_y, wires=i)
            idx += 1
            
    return qml.expval(H_pl)

def compute_delta(theta, beta, n=N_QUBITS):
    # O(1) Analytical Commutator Defect (avoids 4096x4096 matrix multiplication)
    # H = sum c_j P_j. Since P_j are orthogonal, ||H||_F^2 = 2^n sum(c_j^2)
    # [H, T_z2] only picks up the symmetry-breaking terms (which anticommute), adding a factor of 2.
    
    # Calculate coefficients
    idx = 0
    c_zz_sq = 0.0
    for _ in range(n - 1):
        c_zz_sq += theta[idx]**2
        idx += 1
        
    c_sym_sq = 0.0
    c_brk_sq = 0.0
    for _ in range(n):
        c_sym_sq += (theta[idx] * (1 - beta))**2
        c_brk_sq += (theta[idx] * beta)**2
        idx += 1
        
    sum_c_sq = c_zz_sq + c_sym_sq + c_brk_sq
    if sum_c_sq < 1e-12: return 0.0
    
    # delta = 2 * sqrt(sum c_brk^2) / sqrt(sum c_all^2)
    return 2.0 * np.sqrt(c_brk_sq) / (np.sqrt(sum_c_sq) + 1e-15)

# --- 4. VQE Runners ---
def run_vqe_fixed(beta_val, seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    
    def cost(t):
        return circuit_energy(t, beta_val)
        
    history = []
    for step in range(STEPS):
        theta, e = opt.step_and_cost(cost, theta)
        if step % 50 == 0 or step == STEPS - 1:
            state = circuit_state(theta, beta_val)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid)))
            
    final_grad_var = measure_grad_var(theta, beta_val)
    return {'history': history, 'theta': theta, 'beta': beta_val, 'final_grad_var': final_grad_var}

def run_vqe_scheduled(seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    opt = qml.AdamOptimizer(LR)
    
    history = []
    for step in range(STEPS):
        beta_val = step / max(1, (STEPS - 1)) # Ramp 0 to 1
        def cost(t, b=beta_val): return circuit_energy(t, b)
        
        theta, e = opt.step_and_cost(cost, theta)
        if step % 50 == 0 or step == STEPS - 1:
            state = circuit_state(theta, beta_val)
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(e), float(fid), beta_val))
            
    final_grad_var = measure_grad_var(theta, 1.0)
    return {'history': history, 'theta': theta, 'beta': 1.0, 'final_grad_var': final_grad_var}

def run_vqe_learned(lambda_reg, seed):
    np.random.seed(seed)
    theta = np.array(np.random.uniform(-np.pi, np.pi, count_params(N_QUBITS, L_LAYERS)), requires_grad=True)
    beta = np.array(0.5, requires_grad=True) # Start in the middle
    opt = qml.AdamOptimizer(LR)
    
    def cost_with_reg(params):
        t, b = params[0], params[1]
        e = circuit_energy(t, b)
        delta = compute_delta(t, b)
        return e + lambda_reg * delta
        
    # Standard pennylane optimizer needs flat array or tuple of trainable params
    params = (theta, beta)
    
    history = []
    for step in range(STEPS):
        # We compute gradients manually for the custom cost to handle both Pennylane and Numpy parts
        # A simpler approach: alternate updates or use finite diff for beta if needed.
        # Actually, let's use finite differences for beta to keep it robust
        eps = 1e-4
        e = circuit_energy(theta, beta)
        d = compute_delta(theta, beta)
        loss = e + lambda_reg * d
        
        # Grad w.r.t theta (treat beta as a constant scalar for this step)
        g_theta = qml.grad(circuit_energy, argnums=0)(theta, float(beta))
        
        # Grad w.r.t beta via finite differences (includes the regularizer)
        cost_plus = circuit_energy(theta, float(beta + eps)) + lambda_reg * compute_delta(theta, float(beta + eps))
        cost_minus = circuit_energy(theta, float(beta - eps)) + lambda_reg * compute_delta(theta, float(beta - eps))
        g_beta = (cost_plus - cost_minus) / (2 * eps)
        
        theta = theta - LR * g_theta
        beta = np.clip(beta - LR * g_beta, 0.0, 1.0)
        
        if step % 50 == 0 or step == STEPS - 1:
            state = circuit_state(theta, float(beta))
            fid = np.abs(np.vdot(exact_state, state))**2
            history.append((float(loss), float(fid), float(beta)))
            
    final_grad_var = measure_grad_var(theta, float(beta))
    return {'history': history, 'theta': theta, 'beta': float(beta), 'final_grad_var': final_grad_var}

def measure_grad_var(theta, beta, samples=50):
    # Measures the variance of the gradient w.r.t the first parameter at the current point
    # We add a small random perturbation to sample the local variance
    grads = []
    for _ in range(samples):
        pert_theta = np.array(theta + np.random.normal(0, 0.1, len(theta)), requires_grad=True)
        g = qml.grad(circuit_energy, argnums=0)(pert_theta, beta)
        grads.append(float(g[0]))
    return float(np.var(grads))

# --- 5. Main Execution ---
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    res_path = 'results/step4_results.pkl'
    
    if os.path.exists(res_path):
        print(f"Loading checkpoint from {res_path}...")
        with open(res_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}
    
    betas_to_test = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        if seed not in results:
            results[seed] = {}
            
        # S0, S1, S1_1
        for b in tqdm(betas_to_test, desc=f"Fixed Betas (Seed {seed})"):
            k = f'fixed_{b:.1f}'
            if k not in results[seed]:
                res = run_vqe_fixed(b, seed)
                results[seed][k] = res
                with open(res_path, 'wb') as f: pickle.dump(results, f)
            
        # S2
        if 'scheduled' not in results[seed]:
            print("Running S2 (Scheduled)...")
            results[seed]['scheduled'] = run_vqe_scheduled(seed)
            with open(res_path, 'wb') as f: pickle.dump(results, f)
        else:
            print("S2 (Scheduled) already completed. Skipping.")
        
        # S3
        if 'learned' not in results[seed]:
            print("Running S3 (Learned, lambda=0.1)...")
            results[seed]['learned'] = run_vqe_learned(0.1, seed)
            with open(res_path, 'wb') as f: pickle.dump(results, f)
        else:
            print("S3 (Learned) already completed. Skipping.")
        
    print("\n✅ Step 4 Complete. Results saved to results/step4_results.pkl")
