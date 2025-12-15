# GPU Optimization Guide for UCBShift/CSpred

This document describes GPU acceleration opportunities for the UCBShift chemical shift predictor, from quick wins to full model retraining.

## Performance Baseline

| Component | Approx. Time % | GPU Potential |
|-----------|----------------|---------------|
| UCBShift-Y (BLAST + mTM-align) | ~60% | Medium |
| Feature extraction | ~25% | Medium |
| ML model inference (R0/R1/R2) | ~10% | High |
| I/O and data preparation | ~5% | None |

---

## Quick Wins (Low Effort, High Impact)

### 1. ONNX Runtime GPU Inference

Convert existing scikit-learn models to ONNX format for GPU-accelerated inference.

**Speedup:** 2-5x for inference
**Effort:** Low (no retraining needed)

#### One-time Model Conversion

```python
# scripts/convert_to_onnx.py
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import os

ATOMS = ["H", "HA", "C", "CA", "CB", "N"]
MODEL_PATH = "models/"

for atom in ATOMS:
    for level in ["R0", "R1", "R2"]:
        sav_path = f"{MODEL_PATH}{atom}_{level}.sav"
        onnx_path = f"{MODEL_PATH}{atom}_{level}.onnx"

        if not os.path.exists(sav_path):
            continue

        model = joblib.load(sav_path)
        n_features = model.n_features_in_

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Converted {sav_path} -> {onnx_path}")
```

#### Runtime Usage

```python
import onnxruntime as ort

def _load_model_gpu(model_path):
    """Load model with GPU acceleration if available."""
    onnx_path = model_path.replace('.sav', '.onnx')

    if os.path.exists(onnx_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            return ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            pass  # Fall back to sklearn

    return joblib.load(model_path)

def _predict_onnx(session, features):
    """Run prediction with ONNX model."""
    if hasattr(session, 'run'):  # ONNX session
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: features.astype(np.float32)})[0]
    else:  # sklearn model
        return session.predict(features)
```

#### Dependencies

```bash
pip install onnx onnxruntime-gpu skl2onnx
```

---

### 2. Replace BLAST with DIAMOND

DIAMOND is 10,000-20,000x faster than BLAST for protein searches.

**Speedup:** 100x+ for UCBShift-Y
**Effort:** Low

#### Installation

```bash
# Download DIAMOND
wget https://github.com/bbuchfink/diamond/releases/download/v2.1.8/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
mv diamond bins/

# Convert BLAST database to DIAMOND format
bins/diamond makedb --in refDB/refDB.fasta -d refDB/refDB.dmnd
```

#### Code Changes (ucbshifty.py)

```python
DIAMOND_EXE = SCRIPT_PATH + "/bins/diamond"
USE_DIAMOND = os.path.exists(DIAMOND_EXE) and os.path.exists(SCRIPT_PATH + "/refDB/refDB.dmnd")

def blast_diamond(seq, db_name="refDB/refDB.dmnd", working_dir=None):
    """Fast sequence search using DIAMOND."""
    if working_dir is None:
        working_dir = "blast/"
    os.makedirs(working_dir, exist_ok=True)

    # Write query
    fasta_name = working_dir + "query.fasta"
    with open(fasta_name, "w") as f:
        f.write(f">query\n{seq}\n")

    # Run DIAMOND
    out_file = working_dir + "diamond.tsv"
    cmd = f"{DIAMOND_EXE} blastp -d {SCRIPT_PATH}/{db_name} -q {fasta_name} -o {out_file} " \
          f"--very-sensitive --outfmt 6 qseqid sseqid pident length evalue bitscore"
    subprocess.run(cmd, shell=True, capture_output=True)

    # Parse results
    results = {}
    with open(out_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            target = parts[1]
            results[target] = blast_result()
            results[target].target_name = target
            results[target].identity = float(parts[2])
            results[target].Lmatch = int(parts[3])
            results[target].evalue = float(parts[4])
            results[target].bit_score = float(parts[5])

    return results
```

---

### 3. Batch Feature Extraction with Multiprocessing

Parallelize feature extraction across multiple CPU cores.

**Speedup:** 4-8x (scales with cores)
**Effort:** Low

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def extract_features_parallel(pdb_files, n_workers=None):
    """Extract features from multiple PDB files in parallel."""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(pdb_files))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features_list = list(executor.map(build_input, pdb_files))

    return features_list

# In batch mode
if args.batch:
    print(f"Extracting features using {args.worker} workers...")
    all_features = extract_features_parallel(
        [item[0] for item in inputs],
        n_workers=args.worker
    )
```

---

## Medium-Term: GPU-Accelerated Feature Extraction

### Ring Current Calculations (CuPy)

```python
import cupy as cp

def calc_ring_currents_gpu(atom_coords, ring_centers, ring_normals, ring_intensities):
    """
    GPU-accelerated ring current calculation.

    Args:
        atom_coords: (N, 3) array of atom coordinates
        ring_centers: (M, 3) array of aromatic ring centers
        ring_normals: (M, 3) array of ring normal vectors
        ring_intensities: (M,) array of ring intensity factors

    Returns:
        (N,) array of ring current contributions
    """
    # Transfer to GPU
    coords = cp.asarray(atom_coords)
    centers = cp.asarray(ring_centers)
    normals = cp.asarray(ring_normals)
    intensities = cp.asarray(ring_intensities)

    # Vectorized distance calculation: (N, M, 3)
    diff = coords[:, None, :] - centers[None, :, :]
    distances = cp.linalg.norm(diff, axis=2)

    # Avoid division by zero
    distances = cp.maximum(distances, 0.1)

    # Geometric factor: angle between atom-ring vector and ring normal
    cos_angles = cp.einsum('ijk,jk->ij', diff, normals) / (distances * cp.linalg.norm(normals, axis=1))

    # Ring current formula: G = (1 - 3*cos^2(theta)) / r^3
    G = (1 - 3 * cos_angles**2) / (distances**3)

    # Sum contributions from all rings
    ring_currents = cp.sum(G * intensities[None, :], axis=1)

    return cp.asnumpy(ring_currents)
```

### Dihedral Angle Calculations (Vectorized)

```python
def calc_dihedrals_batch_gpu(coords_batch):
    """
    Calculate dihedral angles for multiple residues at once.

    Args:
        coords_batch: (N, 4, 3) array where each row has 4 atom coordinates

    Returns:
        (N,) array of dihedral angles in radians
    """
    coords = cp.asarray(coords_batch)

    # Bond vectors
    b1 = coords[:, 1] - coords[:, 0]
    b2 = coords[:, 2] - coords[:, 1]
    b3 = coords[:, 3] - coords[:, 2]

    # Normal vectors to planes
    n1 = cp.cross(b1, b2)
    n2 = cp.cross(b2, b3)

    # Normalize
    n1 = n1 / cp.linalg.norm(n1, axis=1, keepdims=True)
    n2 = n2 / cp.linalg.norm(n2, axis=1, keepdims=True)
    b2_norm = b2 / cp.linalg.norm(b2, axis=1, keepdims=True)

    # Calculate angle
    m1 = cp.cross(n1, b2_norm)
    x = cp.sum(n1 * n2, axis=1)
    y = cp.sum(m1 * n2, axis=1)

    return cp.asnumpy(cp.arctan2(y, x))
```

---

## Long-Term: PyTorch Model Retraining

### Why Retrain?

1. **GPU inference:** Native CUDA support, 10-100x faster
2. **Batch processing:** Process thousands of residues simultaneously
3. **Modern architecture:** Attention, residual connections
4. **Transfer learning:** Pre-train on large datasets, fine-tune
5. **Uncertainty estimation:** Ensemble or dropout-based confidence

### Architecture Design

```python
import torch
import torch.nn as nn

class UCBShiftNet(nn.Module):
    """
    Neural network replacement for ExtraTrees/RandomForest ensemble.
    Processes all atoms simultaneously with shared feature encoder.
    """

    def __init__(self, n_features, n_atoms=6, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Shared feature encoder
        encoder_layers = []
        in_dim = n_features
        for h_dim in hidden_dims[:-1]:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Atom-specific prediction heads
        self.heads = nn.ModuleDict({
            atom: nn.Sequential(
                nn.Linear(hidden_dims[-2], hidden_dims[-1]),
                nn.GELU(),
                nn.Linear(hidden_dims[-1], 1)
            )
            for atom in ["H", "HA", "C", "CA", "CB", "N"]
        })

    def forward(self, x, atoms=None):
        """
        Args:
            x: (batch_size, n_features) input features
            atoms: List of atoms to predict, or None for all

        Returns:
            dict of {atom: (batch_size,) predictions}
        """
        encoded = self.encoder(x)

        if atoms is None:
            atoms = list(self.heads.keys())

        return {atom: self.heads[atom](encoded).squeeze(-1) for atom in atoms}


class UCBShiftNetWithUncertainty(UCBShiftNet):
    """Extended model with uncertainty estimation via MC Dropout."""

    def predict_with_uncertainty(self, x, n_samples=10):
        self.train()  # Enable dropout

        predictions = {atom: [] for atom in self.heads.keys()}

        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.forward(x)
                for atom, pred in preds.items():
                    predictions[atom].append(pred)

        self.eval()

        results = {}
        for atom in self.heads.keys():
            stacked = torch.stack(predictions[atom])
            results[atom] = {
                'mean': stacked.mean(dim=0),
                'std': stacked.std(dim=0)
            }

        return results
```

### Training Pipeline

```python
# train_model/train_pytorch.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import numpy as np
from pathlib import Path

class ShiftDataset(TensorDataset):
    """Dataset for chemical shift training."""

    def __init__(self, features_df, targets_df, atoms):
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.targets = {
            atom: torch.tensor(targets_df[atom].values, dtype=torch.float32)
            for atom in atoms
        }
        self.masks = {
            atom: ~torch.isnan(self.targets[atom])
            for atom in atoms
        }
        # Replace NaN with 0 for computation (masked out in loss)
        for atom in atoms:
            self.targets[atom][~self.masks[atom]] = 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            {atom: self.targets[atom][idx] for atom in self.targets},
            {atom: self.masks[atom][idx] for atom in self.masks}
        )


def masked_mse_loss(predictions, targets, masks):
    """MSE loss that ignores NaN targets."""
    total_loss = 0
    n_valid = 0

    for atom in predictions:
        mask = masks[atom]
        if mask.sum() > 0:
            pred = predictions[atom][mask]
            tgt = targets[atom][mask]
            total_loss += ((pred - tgt) ** 2).sum()
            n_valid += mask.sum()

    return total_loss / n_valid if n_valid > 0 else torch.tensor(0.0)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for features, targets, masks in loader:
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        masks = {k: v.to(device) for k, v in masks.items()}

        optimizer.zero_grad()
        predictions = model(features)
        loss = masked_mse_loss(predictions, targets, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = {atom: [] for atom in ["H", "HA", "C", "CA", "CB", "N"]}
    all_targets = {atom: [] for atom in ["H", "HA", "C", "CA", "CB", "N"]}
    all_masks = {atom: [] for atom in ["H", "HA", "C", "CA", "CB", "N"]}

    with torch.no_grad():
        for features, targets, masks in loader:
            features = features.to(device)
            predictions = model(features)

            for atom in all_preds:
                all_preds[atom].append(predictions[atom].cpu())
                all_targets[atom].append(targets[atom])
                all_masks[atom].append(masks[atom])

    results = {}
    for atom in all_preds:
        preds = torch.cat(all_preds[atom])
        tgts = torch.cat(all_targets[atom])
        mask = torch.cat(all_masks[atom])

        valid_preds = preds[mask].numpy()
        valid_tgts = tgts[mask].numpy()

        rmse = np.sqrt(np.mean((valid_preds - valid_tgts) ** 2))
        corr = np.corrcoef(valid_preds, valid_tgts)[0, 1]

        results[atom] = {'rmse': rmse, 'corr': corr}

    return results


def main():
    # Configuration
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {DEVICE}")

    # Load data (same preprocessing as original)
    train_df = pd.read_csv("datasets/train_preprocessed.csv")
    val_df = pd.read_csv("datasets/val_preprocessed.csv")

    feature_cols = [c for c in train_df.columns if c not in
                    ["H", "HA", "C", "CA", "CB", "N", "FILE_ID", "RESNAME", "RES_NUM"]]
    target_cols = ["H", "HA", "C", "CA", "CB", "N"]

    train_dataset = ShiftDataset(train_df[feature_cols], train_df[target_cols], target_cols)
    val_dataset = ShiftDataset(val_df[feature_cols], val_df[target_cols], target_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    n_features = len(feature_cols)
    model = UCBShiftNet(n_features).to(DEVICE)

    # Training
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader))

    best_val_rmse = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_results = evaluate(model, val_loader, DEVICE)

        avg_rmse = np.mean([v['rmse'] for v in val_results.values()])

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val RMSE: " + ", ".join(f"{k}={v['rmse']:.3f}" for k, v in val_results.items()))

        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            torch.save(model.state_dict(), "models/ucbshift_pytorch.pt")
            print(f"  Saved best model (avg RMSE: {avg_rmse:.3f})")


if __name__ == "__main__":
    main()
```

### Inference Integration

```python
# Add to CSpred.py

PYTORCH_MODEL_PATH = SCRIPT_PATH + "/models/ucbshift_pytorch.pt"
USE_PYTORCH = os.path.exists(PYTORCH_MODEL_PATH) and torch.cuda.is_available()

def load_pytorch_model():
    """Load PyTorch model for GPU inference."""
    from ucbshift_net import UCBShiftNet

    # Determine n_features from a sample
    n_features = 150  # Set based on training

    model = UCBShiftNet(n_features)
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
    model.eval()
    model.cuda()

    return model

def predict_pytorch(model, features_df):
    """Run prediction using PyTorch model."""
    features = torch.tensor(features_df.values, dtype=torch.float32).cuda()

    with torch.no_grad():
        predictions = model(features)

    return {atom: pred.cpu().numpy() for atom, pred in predictions.items()}
```

---

## Hardware Requirements

### Minimum (Quick Wins)
- Any NVIDIA GPU with CUDA support
- 4GB VRAM
- CUDA 11.0+

### Recommended (Full Pipeline)
- NVIDIA RTX 3060 or better
- 8GB+ VRAM
- CUDA 11.8+
- 32GB RAM for training

### Dependencies

```bash
# Quick wins only
pip install onnxruntime-gpu skl2onnx

# Full GPU pipeline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
pip install onnxruntime-gpu skl2onnx

# DIAMOND (download binary)
wget https://github.com/bbuchfink/diamond/releases/download/v2.1.8/diamond-linux64.tar.gz
```

---

## Benchmarks (Expected)

| Configuration | Single PDB | Batch (100 PDBs) |
|--------------|------------|------------------|
| Original (CPU) | ~30s | ~50min |
| + Model caching | ~25s | ~40min |
| + DIAMOND | ~10s | ~15min |
| + ONNX GPU | ~8s | ~12min |
| + Parallel features | ~8s | ~5min |
| PyTorch GPU (future) | ~3s | ~2min |

---

## Migration Path

1. **Phase 1 (Now):** Implement quick wins
   - ONNX conversion and GPU inference
   - DIAMOND integration
   - Parallel feature extraction

2. **Phase 2 (1-2 weeks):** GPU feature extraction
   - CuPy ring current calculations
   - Vectorized dihedral angles

3. **Phase 3 (2-4 weeks):** PyTorch retraining
   - Prepare training data
   - Train new models
   - Validate against original
   - Deploy hybrid system

4. **Phase 4 (Future):** Advanced features
   - Uncertainty estimation
   - Attention-based architecture
   - Transfer learning from protein language models
