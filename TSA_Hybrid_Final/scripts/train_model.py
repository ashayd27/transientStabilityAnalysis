import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================================================================
# 1. PATH RESOLUTION & DATA LOADING
# ==============================================================================
# Dynamically resolve the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'saved_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print("Loading dynamic physical simulation data (ANDES)...")
try:
    X_raw = np.load(os.path.join(DATA_DIR, 'andes_X_train.npy'))
    y_labels = np.load(os.path.join(DATA_DIR, 'andes_y_train.npy'))
except FileNotFoundError:
    print("Error: Training data not found. Ensure the ANDES generator has finished executing.")
    sys.exit(1)

# The generated data has the shape (Samples, 5, 1000). 
# The 1D-CNN requires the temporal dimension first: (Samples, 1000, 5).
X_train = np.transpose(X_raw, (0, 2, 1))

print(f"Dataset successfully loaded. Input tensor shape: {X_train.shape}")
print(f"Class distribution -> Stable: {np.sum(y_labels == 0)} | Unstable: {np.sum(y_labels == 1)}\n")

# ==============================================================================
# 2. MODEL ARCHITECTURE DESIGN
# ==============================================================================
print("Initializing 1D Convolutional Neural Network architecture...")

model = models.Sequential([
    # Primary Feature Extraction: Identifying fundamental transient spikes
    layers.Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(1000, 5)),
    layers.MaxPooling1D(pool_size=4),

    # Secondary Feature Extraction: Isolating complex divergence patterns
    layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
    layers.MaxPooling1D(pool_size=4),

    # Dimensionality Reduction
    layers.Flatten(),

    # Fully Connected Decision Layer
    layers.Dense(64, activation='relu'),

    # Output Classification Node (Sigmoid for binary probability)
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# ==============================================================================
# 3. TRAINING EXECUTION
# ==============================================================================
print("\nCommencing model training phase...")

# Utilizing an 80/20 validation split to monitor for overfitting during training
history = model.fit(
    X_train, 
    y_labels, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

# ==============================================================================
# 4. EXPORT AND SAVE
# ==============================================================================
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_path = os.path.join(MODEL_DIR, 'cnn_5bus_model.h5')
model.save(model_path)

print(f"\nTraining routine complete. Compiled model securely exported to: {model_path}")