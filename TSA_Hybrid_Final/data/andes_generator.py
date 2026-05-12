import numpy as np
import random
import andes
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import concurrent.futures

andes.config_logger(stream_level=50)

def simulate_single_fault(seed):
    np.random.seed(seed)
    random.seed(seed)
    
    target_buses = [4, 5, 9, 10, 14]
    fault_bus = random.choice(target_buses)
    fault_duration = random.uniform(0.05, 0.45)
    
    case_path = andes.utils.paths.get_case('ieee14/ieee14.json')
    system = andes.load(case_path, setup=False)

    system.add('Fault', idx='F1', bus=fault_bus, tf=1.0, tc=1.0 + fault_duration)
    system.TDS.config.tf = 20.0  
    system.TDS.config.tol = 1e-4
    system.setup()

    try:
        system.PFlow.run()
        system.TDS.run()
    except Exception:
        return None, None 

    t = system.dae.ts.t
    
    delta_indices = [i for i, name in enumerate(system.dae.x_name) if 'delta' in name]
    if len(delta_indices) == 0:
        return None, None
        
    delta = system.dae.ts.x[:, delta_indices]

    delta_rel = delta - delta[:, 0:1]
    
    max_deviation = np.max(np.abs(delta_rel))
    is_unstable = 1 if max_deviation > 3.1415 else 0

    t_fixed = np.arange(0, 20.0, 0.02)
    theta_fixed = np.zeros((5, len(t_fixed)))

    for i in range(5):
        if i < delta_rel.shape[1]:
            interpolator = interp1d(t, delta_rel[:, i], bounds_error=False, fill_value="extrapolate")
            theta_fixed[i] = interpolator(t_fixed)
        
        theta_fixed[i] += np.random.normal(0, 0.01, len(t_fixed))

    return theta_fixed, is_unstable

def build_real_dataset_fast(num_cases=2000):
    print(f"Using ANDES library to simulate {num_cases} physical grid failures")
    X_data, y_labels = [], []
    
    cores = os.cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        results = list(tqdm(executor.map(simulate_single_fault, range(num_cases)), total=num_cases, desc="Simulating Grid Physics"))
        
    for res in results:
        if res[0] is not None:
            X_data.append(res[0])
            y_labels.append(res[1])
            
    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    print(f"\nDataset Built! Final Size: {len(X_data)} valid cases")
    print(f"Distribution -> Stable: {np.sum(y_labels == 0)} | Unstable: {np.sum(y_labels == 1)}")
    
    return X_data, y_labels

if __name__ == "__main__":
    if not os.path.exists('saved_data'): os.makedirs('saved_data')
    
    X, y = build_real_dataset_fast(num_cases=2000)
    
    np.save('saved_data/andes_X_train.npy', X)
    np.save('saved_data/andes_y_train.npy', y)
    print("Physical data saved to 'saved_data' folder")
