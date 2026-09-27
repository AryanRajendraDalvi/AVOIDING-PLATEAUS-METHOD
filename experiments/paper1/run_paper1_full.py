import subprocess
import sys
import time

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"-> {desc}")
    print(f"{'='*60}")
    start = time.time()
    res = subprocess.run([sys.executable, script])
    print(f"Done in {time.time()-start:.1f}s")
    if res.returncode != 0:
        print(f"Error running {script}")
        sys.exit(1)

if __name__ == '__main__':
    run('step4_vqe_optimization.py', 'Step 4: VQE Optimization (Strategies S0-S3)')
    run('step5_metrics.py', 'Step 5: Metrics & Pareto Frontier')
    run('paper1_baselines.py', 'Baselines & Ablations (B3, A1)')
    
    print("\n✅ All rigorous experiments for Paper 1 completed successfully.")
