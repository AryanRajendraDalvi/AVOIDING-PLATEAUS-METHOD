import os
import subprocess
import sys

def run_script(path, cwd=None):
    print(f"\n{'='*70}")
    print(f"Executing: {path}")
    print(f"{'='*70}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [sys.executable, os.path.basename(path)]
    work_dir = cwd if cwd else os.path.dirname(os.path.abspath(path))
    
    result = subprocess.run(cmd, cwd=work_dir, env=env)
    if result.returncode != 0:
        print(f"\n[!] Error executing {path}. Halting pipeline.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    exp_dir = os.path.join(base_dir, "experiments", "paper1")
    
    print("\n" + "#"*70)
    print(" PAPER 1 (AP METHOD) - FULL 7-SEED PIPELINE")
    print("#"*70)
    
    print("\n[!] Force-clearing old PDF checkpoints to ensure complete recalculation...")
    old_files = [
        "paper1_dla_dimension.pdf",
        "paper1_gradient_variance.pdf",
        "paper1_commutator_defect.pdf"
    ]
    for f in old_files:
        fpath = os.path.join(exp_dir, f)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"    Deleted {f}")
            
    # Step 1: DLA Computation (Seed-independent algebra)
    run_script(os.path.join(exp_dir, "step1_dla_computation.py"), cwd=exp_dir)
    
    # Step 2: Gradient Variance at Initialization (Updated to N=7 seeds)
    run_script(os.path.join(exp_dir, "step2_gradient_variance.py"), cwd=exp_dir)
    
    # Step 3: Commutator Defect (Updated to N=7 seeds)
    run_script(os.path.join(exp_dir, "step3_commutator_defect.py"), cwd=exp_dir)
    
    # Step 4 (Scale-up): VQE Optimization with memory-safe sparse matrices
    # Handles n=6, n=12 (7 seeds) and n=16 (12 seeds for pathology)
    run_script(os.path.join(base_dir, "run_phase1_scaleup.py"), cwd=base_dir)
    
    # Step 5 (Metrics): Parses the scale-up checkpoint to generate updated Pareto frontiers
    run_script(os.path.join(base_dir, "step5_scaleup_metrics.py"), cwd=base_dir)
    
    print("\n" + "#"*70)
    print(" PIPELINE COMPLETE! All tables and plots are fully scaled up.")
    print("#"*70)
