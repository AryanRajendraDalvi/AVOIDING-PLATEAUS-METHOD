"""
Paper 1 — Step 5: Metrics & Pareto Frontier
=============================================
Analyzes the results from Step 4 VQE optimization.
Validates Central Hypothesis 4.16: Existence of beta* satisfying
both reachability (F >= 0.90) and trainability (Var >= 10% Var_0).
"""

import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def main():
    res_path = 'results/step4_results.pkl'
    if not os.path.exists(res_path):
        print(f"Error: {res_path} not found. Run Step 4 first.")
        return
        
    with open(res_path, 'rb') as f:
        results = pickle.load(f)
        
    seeds = list(results.keys())
    keys = list(results[seeds[0]].keys())
    
    # Extract fixed beta results
    fixed_keys = [k for k in keys if k.startswith('fixed_')]
    betas = [float(k.split('_')[1]) for k in fixed_keys]
    
    # Sort by beta
    sorted_pairs = sorted(zip(betas, fixed_keys))
    betas = [p[0] for p in sorted_pairs]
    fixed_keys = [p[1] for p in sorted_pairs]
    
    mean_fidelities = []
    mean_grad_vars = []
    
    for k in fixed_keys:
        fids = [results[s][k]['history'][-1][1] for s in seeds]
        vars = [results[s][k]['final_grad_var'] for s in seeds]
        mean_fidelities.append(np.mean(fids))
        mean_grad_vars.append(np.mean(vars))
        
    # Analyze Scheduled (S2) and Learned (S3)
    s2_fids = [results[s]['scheduled']['history'][-1][1] for s in seeds]
    s3_fids = [results[s]['learned']['history'][-1][1] for s in seeds]
    s3_betas = [results[s]['learned']['beta'] for s in seeds]
    
    print("\n" + "="*60)
    print("Hypothesis 4.16 Validation")
    print("="*60)
    
    var_0 = mean_grad_vars[0]
    F_MIN = 0.90
    VAR_THRESH = 0.10 * var_0
    
    print(f"Target Fidelity: >= {F_MIN}")
    print(f"Target Grad Var: >= {VAR_THRESH:.2e} (10% of beta=0 variance: {var_0:.2e})")
    
    valid_betas = []
    for b, f, v in zip(betas, mean_fidelities, mean_grad_vars):
        print(f"  beta={b:.1f} | F={f:.3f} | Var={v:.2e}")
        if f >= F_MIN and v >= VAR_THRESH:
            valid_betas.append((b, f, v))
            
    if valid_betas:
        print(f"\n✅ CONFIRMED: Found {len(valid_betas)} beta* values satisfying both conditions.")
        best = max(valid_betas, key=lambda x: x[1])
        print(f"   Best beta* = {best[0]:.1f} (F={best[1]:.3f})")
    else:
        print("\n❌ REJECTED: No fixed beta satisfied both conditions.")
        
    print(f"\nScheduled (S2) Mean Fidelity: {np.mean(s2_fids):.3f}")
    print(f"Learned (S3) Mean Fidelity: {np.mean(s3_fids):.3f} (Mean final beta: {np.mean(s3_betas):.2f})")
    
    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Fidelity vs Beta
    ax1.plot(betas, mean_fidelities, 'b-o')
    ax1.axhline(F_MIN, color='r', linestyle='--', label='Target F=0.9')
    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Ground State Fidelity')
    ax1.set_title('Reachability')
    ax1.legend()
    
    # 2. Grad Var vs Beta
    ax2.plot(betas, mean_grad_vars, 'g-o')
    ax2.axhline(VAR_THRESH, color='r', linestyle='--', label='10% Var(0)')
    ax2.set_yscale('log')
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Gradient Variance')
    ax2.set_title('Trainability')
    ax2.legend()
    
    # 3. Pareto Frontier
    sc = ax3.scatter(mean_grad_vars, mean_fidelities, c=betas, cmap='viridis', s=100)
    ax3.axhline(F_MIN, color='r', linestyle='--')
    ax3.axvline(VAR_THRESH, color='r', linestyle='--')
    ax3.set_xscale('log')
    ax3.set_xlabel('Gradient Variance')
    ax3.set_ylabel('Fidelity')
    ax3.set_title('Pareto Frontier')
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Beta')
    
    # Add optimal point if exists
    if valid_betas:
        best_b = best[0]
        idx = betas.index(best_b)
        ax3.scatter(mean_grad_vars[idx], mean_fidelities[idx], facecolors='none', edgecolors='red', s=200, lw=2)
        
    plt.tight_layout()
    plt.savefig('paper1_pareto_frontier.pdf', dpi=150, bbox_inches='tight')
    print("\nPlot saved to paper1_pareto_frontier.pdf")

if __name__ == '__main__':
    main()
