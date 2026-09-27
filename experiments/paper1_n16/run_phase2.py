import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import time

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"-> {desc}")
    print(f"{'='*60}")
    start = time.time()
    res = subprocess.run([sys.executable, script])
    print(f"Done in {time.time()-start:.1f}s")
    if res.returncode != 0:
        print(f"Error running {script}. Checkpoint saved, you can resume later.")
        sys.exit(1)

if __name__ == '__main__':
    print("Starting Full Research Pipeline (Estimated Time: 5-10 hours)")
    
    # --- Part A: Theoretical Validation ---
    run('step1_dla_computation.py', 'Step 1: DLA Dimension Computation')
    run('step2_gradient_variance.py', 'Step 2: Gradient Variance Analysis')
    run('step3_commutator_defect.py', 'Step 3: Commutator-Defect Surrogate')
    
    # --- Part B: All Baselines (B0 - B9) ---
    print("\n" + "="*60 + "\n" + "PART B: ALL BASELINES (B0 - B9)".center(60) + "\n" + "="*60)
    # B0, B1, B2, B4 are evaluated as S0, S1, S2 inside step 4
    run('step4_vqe_optimization.py', 'Baselines B0, B1, B2, B4 (VQE Sweeps)')
    run('step5_metrics.py', 'Step 5: Metrics & Pareto Frontier')
    
    # B3 and A1
    run('paper1_baselines.py', 'Baseline B3 and Ablation A1')
    
    # B5, B6, B7, B8, B9
    run('paper1_baselines_extended.py', 'Baselines B5, B6, B7, B8, B9')
    
    # --- Part C: All Ablations (A1 - A6) ---
    print("\n" + "="*60 + "\n" + "PART C: ALL ABLATIONS (A1 - A6)".center(60) + "\n" + "="*60)
    # A1 was evaluated in paper1_baselines.py above
    # A2, A3, A4, A5, A6
    run('paper1_ablations.py', 'Ablations A2, A3, A4, A5, A6')
    
    print("\n✅ All rigorous experiments for Paper 1 completed successfully.")
