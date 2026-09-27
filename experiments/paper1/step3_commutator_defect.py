import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Paper 1 — Step 3: Commutator-Defect Surrogate Δ(θ,β)
=======================================================
Validates Definition 4.13 and Proposition 4.14:

  Δ(θ,β) = (1/m) Σ_k ||[H_θ(β), T_k]||_F / (||H_θ(β)||_F · ||T_k||_F)

Properties to verify:
  - Δ ∈ [0, 2]  always
  - Δ(θ, 0) = 0  always  (H_θ is in commutant at β=0)
  - Δ grows continuously with β
  - Δ is differentiable in β (suitable as a regularizer for Learned-β strategy)

Run: python step3_commutator_defect.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Pauli utilities ──────────────────────────────────────────────────────────

PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0,1],[1,0]], dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'Z': np.array([[1,0],[0,-1]], dtype=complex),
}

def pauli_matrix(string):
    result = np.array([[1.+0j]])
    for c in string:
        result = np.kron(result, PAULI[c])
    return result

def frob_norm(A):
    return float(np.sqrt(np.trace(A.conj().T @ A).real))

def commutator(A, B):
    return A @ B - B @ A

# ─── Z2 generator ────────────────────────────────────────────────────────────

def z2_generator(n):
    """U_g = ⊗_i X_i  (global spin flip, Z2 symmetry of TFIM)"""
    X = PAULI['X']
    result = np.array([[1.+0j]])
    for _ in range(n):
        result = np.kron(result, X)
    return result

# ─── Build H_θ(β) ────────────────────────────────────────────────────────────

def build_H_theta(theta, beta, n):
    """
    Effective Hamiltonian:
      H_θ(β) = Σ_{i<n-1} θ_i · ZZ_i  +  Σ_i θ_{n-1+i} · [(1-β)X_i + β·Z_i]
    """
    H = np.zeros((2**n, 2**n), dtype=complex)
    idx = 0

    # ZZ symmetric terms
    for i in range(n - 1):
        ZZ = pauli_matrix('I' * i + 'ZZ' + 'I' * (n - i - 2))
        H += theta[idx] * ZZ
        idx += 1

    # Interpolated single-qubit terms
    for i in range(n):
        X_op = pauli_matrix('I' * i + 'X' + 'I' * (n - i - 1))
        Z_op = pauli_matrix('I' * i + 'Z' + 'I' * (n - i - 1))
        H += theta[idx] * ((1 - beta) * X_op + beta * Z_op)
        idx += 1

    return H

# ─── Commutator-defect Δ ─────────────────────────────────────────────────────

def commutator_defect(H_theta, symmetry_generators):
    """
    Δ(θ,β) = (1/m) Σ_k ||[H_θ, T_k]||_F / (||H_θ||_F · ||T_k||_F)
    Returns 0 if H_theta ≈ 0 (convention from paper).
    """
    norm_H = frob_norm(H_theta)
    if norm_H < 1e-12:
        return 0.0

    m = len(symmetry_generators)
    total = 0.0
    for T in symmetry_generators:
        comm = commutator(H_theta, T)
        numerator = frob_norm(comm)
        denominator = norm_H * frob_norm(T)
        total += numerator / (denominator + 1e-15)

    return total / m


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_defect_sweep(n=6, N_random=200, seeds=[42, 43, 44, 45, 46, 47, 48]):
    n_params = (n - 1) + n   # ZZ terms + single-qubit terms
    T_z2 = z2_generator(n)
    betas = np.linspace(0.0, 1.0, 25)

    print(f"\n{'='*60}")
    print(f"Commutator-Defect Surrogate | n={n} | N_seeds={len(seeds)} | N_rand={N_random}")
    print(f"{'='*60}")

    mean_deltas, std_deltas = [], []

    for beta in betas:
        all_deltas = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for _ in range(N_random):
                theta = rng.normal(0, 1, n_params)
                H = build_H_theta(theta, beta, n)
                delta = commutator_defect(H, [T_z2])
                all_deltas.append(delta)
                
        m = float(np.mean(all_deltas))
        s = float(np.std(all_deltas))
        mean_deltas.append(m)
        std_deltas.append(s)

    # ─── Validate Proposition 4.14 ────────────────────────────────────────────
    print()
    print("beta=0.00:  Delta should be exactly 0 (H in commutant)")
    print(f"         Delta measured = {mean_deltas[0]:.2e}  (expect < 1e-10)")
    print()
    print("beta=1.00:  Delta should be > 0 (H not in commutant)")
    print(f"         Delta measured = {mean_deltas[-1]:.4f}")
    
    # ─── Print table ──────────────────────────────────────────────────────────
    print()
    print(f"{'Beta':>6}  {'Mean Delta':>10}  {'Std Delta':>10}")
    for i, beta in enumerate(betas):
        print(f"{beta:6.2f}  {mean_deltas[i]:10.4f}  {std_deltas[i]:10.4f}")

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(betas,
                    np.array(mean_deltas) - np.array(std_deltas),
                    np.array(mean_deltas) + np.array(std_deltas),
                    alpha=0.3, color='blue', label='+/- 1 std')
    ax.plot(betas, mean_deltas, 'b-o', markersize=4, label='Mean Delta(beta)')
    ax.set_xlabel('Beta', fontsize=12)
    ax.set_ylabel('Commutator-Defect Delta', fontsize=12)
    ax.set_title(f'Commutator-Defect Surrogate (n={n}, N_seeds={len(seeds)})', fontsize=13)
    ax.set_ylim([-0.05, max(mean_deltas) * 1.2])
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('paper1_commutator_defect_n6.pdf', dpi=150, bbox_inches='tight')
    print("\nFigure saved: paper1_commutator_defect_n6.pdf")

    return betas, mean_deltas, std_deltas


if __name__ == "__main__":
    import os
    if os.path.exists("paper1_commutator_defect_n6.pdf"):
        print("Loading checkpoint... Step 3 already complete (PDF exists). Skipping.")
    else:
        run_defect_sweep(n=6, N_random=300)
