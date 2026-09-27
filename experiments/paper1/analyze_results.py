import pickle
import numpy as np
import sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

seeds = [42, 43, 44]

print("=" * 80)
print("COMPREHENSIVE N=12 RESULTS ANALYSIS")
print("=" * 80)

# --- 1. Inspect step4 structure ---
print("\n>>> Inspecting step4_results.pkl key structure...")
with open('results/step4_results.pkl', 'rb') as f:
    step4 = pickle.load(f)

for s in seeds:
    print(f"\nSeed {s} keys: {list(step4[s].keys())}")
    # Show first key's structure
    for k in step4[s]:
        v = step4[s][k]
        if isinstance(v, dict):
            print(f"  '{k}' -> dict with keys: {list(v.keys())}")
            if 'history' in v:
                hist = v['history']
                if hist:
                    print(f"    history[-1] = {hist[-1]}")
        else:
            print(f"  '{k}' -> {type(v).__name__}: {v}")
        break  # just first key per seed

# --- 2. Inspect baselines_results.pkl ---
print("\n>>> Inspecting baselines_results.pkl...")
with open('results/baselines_results.pkl', 'rb') as f:
    baselines = pickle.load(f)
print(f"Top-level keys: {list(baselines.keys())}")
for k in baselines:
    if isinstance(baselines[k], dict):
        print(f"  '{k}' -> {list(baselines[k].keys())}")
        for kk in baselines[k]:
            print(f"    '{kk}' -> {type(baselines[k][kk]).__name__}: {repr(baselines[k][kk])[:200]}")
    else:
        print(f"  '{k}' -> {type(baselines[k]).__name__}: {repr(baselines[k])[:200]}")

# --- 3. Inspect ablations_results.pkl ---
print("\n>>> Inspecting ablations_results.pkl...")
with open('results/ablations_results.pkl', 'rb') as f:
    ablations = pickle.load(f)
print(f"Top-level keys: {list(ablations.keys())}")
for k in ablations:
    if isinstance(ablations[k], dict):
        print(f"  '{k}' -> {list(ablations[k].keys())}")
        for kk in ablations[k]:
            val = ablations[k][kk]
            print(f"    '{kk}' -> {type(val).__name__}: {repr(val)[:200]}")
    else:
        print(f"  '{k}' -> {type(ablations[k]).__name__}: {repr(ablations[k])[:200]}")

# --- 4. Now do the actual analysis with correct keys ---
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)

# Step4 fixed betas
print("\n--- Fixed Beta Sweeps ---")
betas_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
print(f"{'Beta':<8} {'Mean Energy':>14} {'Mean Fidelity':>14} {'Std Fidelity':>13}")
print("-" * 52)

for b in betas_list:
    energies = []
    fidelities = []
    for s in seeds:
        # Try different key formats
        for key_fmt in [f'beta_{b}', f'beta_{b:.1f}', b, str(b)]:
            if key_fmt in step4[s]:
                data = step4[s][key_fmt]
                if isinstance(data, dict) and 'history' in data:
                    hist = data['history']
                    if hist:
                        last = hist[-1]
                        energies.append(last[0])
                        fidelities.append(last[1])
                break
    
    if fidelities:
        label = ""
        if b == 0.0: label = " <-- B1 (Symmetric)"
        elif b == 1.0: label = " <-- B2 (Unrestricted)"
        else: label = " <-- PROPOSED"
        print(f"b={b:<5} {np.mean(energies):>14.6f} {np.mean(fidelities):>14.4f} {np.std(fidelities):>13.4f}{label}")
    else:
        print(f"b={b:<5}  -- NO DATA FOUND --")

# S2
print("\n--- S2 (Scheduled Beta Ramp) ---")
s2_fids = []
for s in seeds:
    for key in ['scheduled', 'S2', 's2']:
        if key in step4[s]:
            data = step4[s][key]
            if isinstance(data, dict) and 'history' in data:
                hist = data['history']
                if hist:
                    s2_fids.append(hist[-1][1])
                    print(f"  Seed {s}: Fidelity = {hist[-1][1]:.4f}, Energy = {hist[-1][0]:.6f}")
            break
if s2_fids:
    print(f"  MEAN: {np.mean(s2_fids):.4f} +/- {np.std(s2_fids):.4f}")

# S3
print("\n--- S3 (Learned Beta) ---")
s3_fids = []
for s in seeds:
    for key in ['learned_0.1', 'S3', 's3', 'learned']:
        if key in step4[s]:
            data = step4[s][key]
            if isinstance(data, dict) and 'history' in data:
                hist = data['history']
                if hist:
                    s3_fids.append(hist[-1][1])
                    beta_val = data.get('beta', 'N/A')
                    print(f"  Seed {s}: Fidelity = {hist[-1][1]:.4f}, Final Beta = {beta_val}")
            break
if s3_fids:
    print(f"  MEAN: {np.mean(s3_fids):.4f} +/- {np.std(s3_fids):.4f}")

# Extended baselines
print("\n--- Extended Baselines B5-B9 ---")
with open('results/baselines_extended_results.pkl', 'rb') as f:
    ext = pickle.load(f)

for bname in ['B5', 'B6', 'B7', 'B8', 'B9']:
    fids = []
    ens = []
    for s in seeds:
        if s in ext and bname in ext[s]:
            r = ext[s][bname]
            if isinstance(r, tuple):
                ens.append(r[0])
                fids.append(r[1])
    labels = {'B5': 'Horizontal Gates', 'B6': 'Standard HEA', 'B7': 'QAOA', 
              'B8': 'ADAPT-VQE', 'B9': 'ADAPT-VQE+Delta'}
    if fids:
        print(f"  {bname} ({labels[bname]}): F={np.mean(fids):.4f} +/- {np.std(fids):.4f}, E={np.mean(ens):.6f}")
    else:
        print(f"  {bname} ({labels[bname]}): NO DATA")

# Ablations
print("\n--- Ablations A2-A6 ---")
with open('results/ablations_results.pkl', 'rb') as f:
    abl = pickle.load(f)

for aname in ['A2', 'A3_Z', 'A3_Y', 'A3_ZZ', 'A4', 'A5', 'A6']:
    ens = []
    fids = []
    for s in seeds:
        if s in abl and aname in abl[s]:
            r = abl[s][aname]
            if isinstance(r, tuple):
                ens.append(r[0])
                if len(r) >= 2: fids.append(r[1])
            elif isinstance(r, (float, int, np.floating)):
                ens.append(float(r))
    labels = {'A2': 'SGD Optimizer', 'A3_Z': 'Gen Z only', 'A3_Y': 'Gen Y only', 
              'A3_ZZ': 'Gen ZZ only', 'A4': 'Exp Schedule', 'A5': 'Shot Noise', 'A6': 'U(1) Sym'}
    if fids:
        print(f"  {aname} ({labels[aname]}): F={np.mean(fids):.4f} +/- {np.std(fids):.4f}, E={np.mean(ens):.6f}")
    elif ens:
        print(f"  {aname} ({labels[aname]}): E={np.mean(ens):.6f} +/- {np.std(ens):.6f} (no fidelity stored)")
    else:
        print(f"  {aname} ({labels[aname]}): NO DATA")

# B3 and A1
print("\n--- B3 and A1 ---")
with open('results/baselines_results.pkl', 'rb') as f:
    bas = pickle.load(f)
for bname in ['B3', 'A1']:
    fids = []
    ens = []
    for s in seeds:
        if s in bas and bname in bas[s]:
            r = bas[s][bname]
            if isinstance(r, tuple):
                ens.append(r[0])
                if len(r) >= 2: fids.append(r[1])
    if fids:
        print(f"  {bname}: F={np.mean(fids):.4f} +/- {np.std(fids):.4f}, E={np.mean(ens):.6f}")
    elif ens:
        print(f"  {bname}: E={np.mean(ens):.6f} (no fidelity)")
    else:
        print(f"  {bname}: NO DATA FOUND")

print("\n\nExact Ground State Energy: -13.031931")
print("DONE.")
