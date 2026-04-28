import numpy as np
from scipy import signal  # <--- THIS WAS THE MISSING PIECE
from sklearn.linear_model import LinearRegression

def get_all_generator_thetas(raw_data):
    """
    Assumes the first 5 rows of your 14x1000 data are the generators.
    Calculates Center of Inertia (COI) and returns relative angles.
    """
    # Assuming rows 0-4 are the 5 generators in the 14-bus system
    thetas = raw_data[:5, :] 
    coi = np.mean(thetas, axis=0)
    return thetas - coi

def estimate_mle_robust(x, m=5, J=10):
    """
    Tuned for 14-Bus dynamics:
    Uses Linear Regression for a stable slope instead of jumpy snapshots.
    """
    N = len(x)
    N_eff = N - (m - 1) * J
    if N_eff < 100: return -0.5
    
    # 1. Phase Space Embedding
    X = np.column_stack([x[i*J : i*J + N_eff] for i in range(m)])
    
    # 2. Track divergence growth
    lookahead = 30
    distances = []
    
    # Sample 50 points to see how the trajectory 'stretches'
    for t in range(50):
        d0 = np.linalg.norm(X[t] - X[t+1])
        dt = np.linalg.norm(X[t+lookahead] - X[t+1+lookahead])
        if d0 > 1e-9:
            distances.append(np.log(dt / d0))
            
    if not distances: return -0.5
    
    # 3. Regression for robust slope calculation
    y = np.array(distances).reshape(-1, 1)
    x_axis = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x_axis, y)
    
    return float(model.coef_[0][0])

def compute_system_mle(raw_case):
    """
    THE SCANNER:
    Applies the polynomial filter to all 5 generators and returns 
    the maximum MLE found in the system.
    """
    gen_thetas = get_all_generator_thetas(raw_case)
    mle_values = []
    
    for i in range(5):
        # Smoothing out the PMU noise (The Savior!)
        polished = signal.savgol_filter(gen_thetas[i], 51, 3)
        mle_values.append(estimate_mle_robust(polished))
        
    return max(mle_values)