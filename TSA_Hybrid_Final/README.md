# Physics-Informed Hybrid Triage for Real-Time Transient Stability Assessment

This repository contains the codebase for a novel Hybrid Transient Stability Assessment (TSA) framework. By integrating a physics-based Maximum Lyapunov Exponent (MLE) scanner with a data-driven 1D Convolutional Neural Network (CNN), this system achieves sub-cycle fault detection speeds on the **IEEE 14-bus standard test system**.

**Key Performance Metrics (10,000-Case Benchmark):**
* **Accuracy:** 99.99%
* **Average Latency:** 12.26 ms 
* **Computational Reduction:** ~46% of cases resolved by pure physics, bypassing AI inference latency.

---

## Repository Structure

* `/data` - Contains the ANDES-based synthetic data generator (`andes_generator.py`).
* `/src` - The core physics engine (`physics.py`) for multi-machine MLE scanning.
* `/models` - Pre-trained deep learning models (e.g., `cnn_14bus_model.h5`).
* `/scripts` - Execution scripts for probing, benchmarking, and generating publication plots.
* `requirements.txt` - Python package dependencies.

*(Note: The `saved_data/` directory is ignored via `.gitignore` as it contains massive high-resolution `.npy` arrays generated during the simulation phase.)*

---

## Installation & Prerequisites

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/TSA_Hybrid_Final.git](https://github.com/YOUR_USERNAME/TSA_Hybrid_Final.git)
cd TSA_Hybrid_Final
```

**2. Install Python dependencies:**
The project relies on standard scientific libraries and the ANDES simulation engine.
```bash
pip install -r requirements.txt
```
## How to Run the Project

**1. Generate the Physical Grid Data**
Because the physical simulation data is too large to host on GitHub, you must generate it locally. This script uses multi-processing to simulate thousands of dynamic grid faults via the ANDES engine and injects realistic PMU thermal noise (σ = 0.01 rad).
```bash
python data/andes_generator.py
```

**2. Discover Triage Thresholds (using Lambda Probing)**
Run the statistical probe to map the probability density functions of the Stable and Unstable cases. This script performs a high-volume analysis of the generated transients to mathematically define the "Grey Zone" boundaries ($0.018 \leq \lambda \leq 0.052$) used by the Hybrid logic.
```bash
python scripts/probe_lambdas.py
```

**3. Run the Performance Benchmark**
This is the final execution step. The script streams the generated data sequentially to calculate the real-time performance of all three methodologies. It measures accuracy and latency to verify the hybrid framework's ability to resolve cases at sub-cycle speeds (~12.26 ms).
```bash
python scripts/run_benchmark.py
```
