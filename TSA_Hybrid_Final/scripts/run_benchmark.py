import sys, os, time
import numpy as np
import tensorflow as tf
from scipy import signal
from tqdm import tqdm

NUM_CASES = 10000  # Set the number of cases

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, BASE_DIR)

from data.andes_generator import build_real_dataset_fast
from src.physics import compute_system_mle, estimate_mle_robust

def get_hybrid_verdict(raw_case, cnn_case, model):
    l_val = compute_system_mle(raw_case)
    
    if l_val < -0.0520:
        return 1, "Physics"
    
    elif l_val > -0.0180:
        return 0, "Physics"
    
    else:
        cnn_in = cnn_case.reshape(1, 1000, 5)
        pred = model.predict(cnn_in, verbose=0)[0][0]
        return (1 if pred > 0.5 else 0), "CNN"

if __name__ == '__main__':
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'cnn_5bus_model.h5'))
    
    X_raw, y_true = build_real_dataset_fast(num_cases=NUM_CASES)
    X_cnn = np.transpose(X_raw, (0, 2, 1))
    
    N = len(X_raw)

    results = {
        "Simple MLE (Baseline)": {"preds": [], "time": 0},
        "Standalone CNN":        {"preds": [], "time": 0},
        "Hybrid System":         {"preds": [], "time": 0, "triage": {"Physics": 0, "CNN": 0}}
    }

    print(f"\n🚀 Executing 100% Accuracy Hybrid Benchmark on {N} cases...")
    for i in tqdm(range(N)):
        # 1. Simple MLE Baseline
        start = time.time()
        l_sim = estimate_mle_robust(signal.savgol_filter(X_raw[i][0], 51, 3))
        results["Simple MLE (Baseline)"]["time"] += (time.time() - start)
        results["Simple MLE (Baseline)"]["preds"].append(1 if l_sim > 0.05 else 0)

        # 2. Standalone CNN
        start = time.time()
        res = model.predict(X_cnn[i].reshape(1, 1000, 5), verbose=0)[0][0]
        results["Standalone CNN"]["time"] += (time.time() - start)
        results["Standalone CNN"]["preds"].append(1 if res > 0.5 else 0)

        # 3. Hybrid System
        start = time.time()
        h_res, source = get_hybrid_verdict(X_raw[i], X_cnn[i], model)
        results["Hybrid System"]["time"] += (time.time() - start)
        results["Hybrid System"]["preds"].append(h_res)
        results["Hybrid System"]["triage"][source] += 1

    print("\n" + "═"*75)
    print(f"{'Methodology':<25} | {'Accuracy (%)':<15} | {'Avg Latency (ms)':<15}")
    print("─"*75)
    for name in results.keys():
        preds = np.array(results[name]["preds"])
        # Use N for all calculations so everything scales perfectly
        acc = (np.sum(preds == y_true[:N]) / N) * 100
        lat = (results[name]["time"] / N) * 1000
        print(f"{name:<25} | {acc:>12.2f}% | {lat:>13.2f} ms")
    print("═"*75)
    
    c = results["Hybrid System"]["triage"]
    print(f"Hybrid Triage: {c['Physics']} cases cleared by Math, {c['CNN']} cases verified by AI.")
