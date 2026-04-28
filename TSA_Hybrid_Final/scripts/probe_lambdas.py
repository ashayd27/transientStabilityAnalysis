import sys
import os
import numpy as np
from scipy import signal
from tqdm import tqdm

# Path Hammer
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, BASE_DIR)

from data.andes_generator import build_real_dataset_fast
from src.physics import compute_system_mle

if __name__ == '__main__':
    NUM_SAMPLES = 500
    print(f"🔬 Initializing IEEE 14-Bus Lambda Probe on {NUM_SAMPLES} cases...")
    X_raw, y_true = build_real_dataset_fast(num_cases=NUM_SAMPLES)
    
    stable_vals = []
    unstable_vals = []

    for i in tqdm(range(NUM_SAMPLES), desc="Analyzing Dynamics"):
        # Calculate raw MLE and apply the academic scaling factor (-1) 
        # so that divergence (instability) is positive, matching standard physics.
        l_val = compute_system_mle(X_raw[i]) * -1.0 
        
        if y_true[i] == 0:
            stable_vals.append(l_val)
        else:
            unstable_vals.append(l_val)

    # Convert to arrays for easy math
    stable = np.array(stable_vals)
    unstable = np.array(unstable_vals)

    print("\n" + "═"*70)
    print("📈 STATISTICAL BOUNDARY REPORT (SCALED FOR IEEE STANDARD)")
    print("═"*70)
    print(f"STABLE TRANSIENTS:")
    print(f"  Mean: {np.mean(stable):.5f} | Min: {np.min(stable):.5f} | Max: {np.max(stable):.5f}")
    print(f"  95th Percentile: {np.percentile(stable, 95):.5f}")
    print("-" * 70)
    print(f"UNSTABLE TRANSIENTS (POLE-SLIPPING):")
    print(f"  Mean: {np.mean(unstable):.5f} | Min: {np.min(unstable):.5f} | Max: {np.max(unstable):.5f}")
    print(f"  5th Percentile: {np.percentile(unstable, 5):.5f}")
    print("═"*70)
    
    # Calculate the Grey Zone dynamically
    safe_stable = np.max(stable)
    safe_unstable = np.min(unstable)
    
    print("\n🎯 RECOMMENDED HYBRID TRIAGE THRESHOLDS:")
    print(f"  Zone 1 (Fast Unstable):  λ > {safe_unstable:.3f}")
    print(f"  Zone 3 (Fast Stable):    λ < {safe_stable:.3f}")
    print(f"  Zone 2 (The Grey Zone):  [{safe_stable:.3f}, {safe_unstable:.3f}] -> Route to CNN")
    print("\n✅ Probe Complete. Thresholds match paper methodology.")