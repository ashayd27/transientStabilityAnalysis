import numpy as np
import random
import andes
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import concurrent.futures

# Mute ANDES so 8 different CPU cores don't scream at you in the terminal at once
andes.config_logger(stream_level=50)

def simulate_single_fault(seed):
    """The worker function: One CPU core simulates exactly one grid fault."""
    # We must give each worker its own seed so they don't simulate the exact same fault!
    np.random.seed(seed)
    random.seed(seed)
    
    # We target the non-generator 'neighborhood' buses to see how the 5 power plants react
    target_buses = [4, 5, 9, 10, 14]
    fault_bus = random.choice(target_buses)
    fault_duration = random.uniform(0.05, 0.45)
    
    # Load the real IEEE 14-bus map
    case_path = andes.utils.paths.get_case('ieee14/ieee14.json')
    system = andes.load(case_path, setup=False)

    # Throw the wrench into the gears
    system.add('Fault', idx='F1', bus=fault_bus, tf=1.0, tc=1.0 + fault_duration)
    system.TDS.config.tf = 20.0  
    system.TDS.config.tol = 1e-4
    system.setup()

    try:
        # Run the physics
        system.PFlow.run()
        system.TDS.run()
    except Exception:
        return None, None 

    t = system.dae.ts.t
    
    # The Magic Trick: Find the 5 Power Plants (Generators) hidden in the 14-bus city
    delta_indices = [i for i, name in enumerate(system.dae.x_name) if 'delta' in name]
    if len(delta_indices) == 0:
        return None, None
        
    delta = system.dae.ts.x[:, delta_indices]

    # Calculate relative phase angles (everyone compared to Generator 1)
    delta_rel = delta - delta[:, 0:1]
    
    # The Physics Verdict: Did they swing past 180 degrees (3.14 radians)?
    max_deviation = np.max(np.abs(delta_rel))
    is_unstable = 1 if max_deviation > 3.1415 else 0

    # Format the audio for the CNN (Exactly 20 seconds, 50Hz = 1000 samples)
    t_fixed = np.arange(0, 20.0, 0.02)
    theta_fixed = np.zeros((5, len(t_fixed)))

    for i in range(5):
        if i < delta_rel.shape[1]:
            # Interpolate the messy physics time-steps into our perfectly clean 50Hz track
            interpolator = interp1d(t, delta_rel[:, i], bounds_error=False, fill_value="extrapolate")
            theta_fixed[i] = interpolator(t_fixed)
        
        # Add the 'Rust' (Standard PMU thermal noise)
        theta_fixed[i] += np.random.normal(0, 0.01, len(t_fixed))

    return theta_fixed, is_unstable

def build_real_dataset_fast(num_cases=2000):
    print(f"⚡ Booting ANDES Physics Engine. Simulating {num_cases} physical grid failures...")
    X_data, y_labels = [], []
    
    # Ask your Mac exactly how many CPU cores it has available
    cores = os.cpu_count()
    print(f"🚀 Firing up all {cores} CPU cores for Multiprocessing...")
    
    # Launch the multi-threaded construction crew!
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # 'map' hands out the 2000 jobs to your available CPU cores and tracks the progress
        results = list(tqdm(executor.map(simulate_single_fault, range(num_cases)), total=num_cases, desc="Simulating Grid Physics"))
        
    for res in results:
        if res[0] is not None: # If the grid didn't completely explode into unreadable math
            X_data.append(res[0])
            y_labels.append(res[1])
            
    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    print(f"\n✅ Dataset Built! Final Size: {len(X_data)} valid cases.")
    print(f"Distribution -> Stable: {np.sum(y_labels == 0)} | Unstable: {np.sum(y_labels == 1)}")
    
    return X_data, y_labels

if __name__ == "__main__":
    # Ensure the vault exists
    if not os.path.exists('saved_data'): os.makedirs('saved_data')
    
    # The multi-core engine will chew through this in a fraction of the time
    X, y = build_real_dataset_fast(num_cases=2000)
    
    # Save the physical data safely into the vault
    np.save('saved_data/andes_X_train.npy', X)
    np.save('saved_data/andes_y_train.npy', y)
    print("💾 Physical data saved to 'saved_data' folder.")