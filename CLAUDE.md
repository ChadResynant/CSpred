# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UCBShift (CSpred) is a protein NMR chemical shift predictor for backbone atoms (H, HA, C, CA, CB, N) in aqueous solution. It combines two prediction modules:

- **UCBShift-X**: Machine learning module using ensemble decision trees (ExtraTrees + RandomForest) with structural features extracted from PDB files
- **UCBShift-Y**: Transfer prediction module using BLAST sequence alignment and mTM-align structure alignment to transfer shifts from reference proteins

The combined UCBShift model achieves state-of-the-art accuracy by integrating both approaches.

## Quick Start

### Prediction (Single PDB)
```bash
python CSpred.py your_protein.pdb
```
Outputs `shifts.csv` with predictions for all backbone atoms.

### Prediction Options
```bash
python CSpred.py input.pdb -o output.csv     # Custom output name
python CSpred.py input.pdb -pH 7.0           # Specify pH (default: 5)
python CSpred.py input.pdb -x                # UCBShift-X only (ML only, faster)
python CSpred.py input.pdb -y                # UCBShift-Y only (transfer only)
python CSpred.py input.pdb -t                # Test mode (excludes near-identical refs)
```

### Batch Prediction
```bash
python CSpred.py batch_list.txt -b -o output_dir/ -w 4
```
Where `batch_list.txt` contains PDB paths (optionally with pH values).

### UCBShift-Y Standalone
```bash
python ucbshifty.py input.pdb -s 1           # Normal strictness
python ucbshifty.py input.pdb -s 0           # Strict (exact residue match only)
python ucbshifty.py input.pdb -s 2           # Permissive (transfer all)
python ucbshifty.py input.pdb -2             # Output secondary shifts
python ucbshifty.py input.pdb -y             # SHIFTY mode (top BLAST hit only)
```

## Model Files

Trained models (18 `.sav` files) must be downloaded from Dryad and extracted to `models/`:
```bash
tar -xzf models.tgz  # Creates models/*.sav
```

Each atom type has three model levels:
- `{ATOM}_R0.sav`: Base ExtraTrees predictor (structural features only)
- `{ATOM}_R1.sav`: RandomForest with R0 predictions added (UCBShift-X)
- `{ATOM}_R2.sav`: RandomForest combining R1 + UCBShift-Y predictions

## Architecture

### Prediction Pipeline (`CSpred.py`)
1. `build_input()` → Extract SPARTA+-style features via `spartap_features.py`
2. `data_preprocessing()` → Feature engineering (power transforms, hydrophobicity)
3. For each atom type:
   - Load R0, generate base predictions
   - Add R0 predictions as features, run R1 (UCBShift-X result)
   - If UCBShift-Y available, combine and run R2 (final UCBShift)
4. Add random coil shifts back for absolute chemical shift values

### Feature Extraction (`spartap_features.py`)
`PDB_SPARTAp_DataReader` class extracts per-residue features:
- Phi/psi/chi dihedral angles (cos/sin)
- Hydrogen bond geometry (distance, angles, energy)
- S2 order parameters (contact model)
- BLOSUM62 substitution scores
- Ring current contributions (aromatic residues)
- DSSP secondary structure
- Half-sphere exposure (HSE)
- B-factors

Features include i-1, i, i+1 tripeptide context.

### Transfer Prediction (`ucbshifty.py`)
1. BLAST sequence search against refDB
2. mTM-align structure alignment for BLAST hits
3. Needleman-Wunsch alignment refinement
4. Weighted shift transfer with BLOSUM62 substitution scores
5. TM-score weighted averaging across references

### Training Pipeline (`train_model/`)
1. `download_pdbs.py`: Download and hydrogenate PDB files
2. `build_df.py`: Generate feature CSVs from PDBs
3. `make_Y_preds.py`: Generate UCBShift-Y predictions for training set
4. `train.py`: Train R0/R1/R2 models with K-fold cross-validation
5. `evaluate.py`: Evaluate on test set

## Key Dependencies

- **Python >=3.5**
- **Biopython 1.74** (exact version required for BLOSUM62 compatibility)
- **scikit-learn 0.22**
- **External programs**: BLAST 2.9.0, mTM-align, DSSP, REDUCE (for hydrogenation)

## Data Layout

```
CSpred/
├── CSpred.py              # Main prediction script
├── ucbshifty.py           # UCBShift-Y standalone
├── spartap_features.py    # Feature extraction
├── data_prep_functions.py # Feature engineering utilities
├── toolbox.py             # Utility functions, random coil values
├── models/                # Trained model files (.sav)
├── refDB/                 # Reference database for UCBShift-Y
│   ├── *.blastdb          # BLAST databases
│   ├── pdbs/              # Reference PDB structures
│   └── shifts_df/         # Reference chemical shifts (CSV)
├── bins/                  # External binaries (BLAST, mTM-align)
└── train_model/           # Training scripts and data
```

## Output Format

The output CSV contains:
- `RESNUM`, `RESNAME`: Residue identifier
- `{ATOM}_X`: UCBShift-X prediction
- `{ATOM}_Y`: UCBShift-Y prediction
- `{ATOM}_UCBShift`: Combined final prediction
- `{ATOM}_BEST_REF_SCORE/COV/MATCH`: Quality metrics for Y predictions

## Common Issues

- **Biopython version**: Must use exactly 1.74; newer versions break BLOSUM62 matrix handling
- **Missing hydrogens**: Structures without H atoms significantly degrade prediction quality; use REDUCE
- **Extreme pH**: Predictions outside pH 2-12 may be unreliable
- **Multi-chain PDBs**: Only first chain is processed; split multi-chain files

## Atom Types

`toolbox.ATOMS = ["H", "HA", "C", "CA", "CB", "N"]`

All chemical shifts are backbone atoms plus beta carbon.

---

## Code Patterns and Conventions

### Path Resolution
All scripts use `SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))` to resolve paths relative to the script location, not the working directory. This allows the code to work when called from any location.

### NaN Handling
The codebase extensively uses `np.nan` for missing chemical shift values:
- Missing predictions are set to NaN, not zero
- NaN propagates through calculations
- Use `.notnull()` to filter valid predictions

### Random Coil Shifts
Chemical shifts are predicted as **secondary shifts** (deviation from random coil) internally, then random coil values are added back:
```python
# Internal: predict secondary shift
secondary_shift = model.predict(features)
# Output: absolute shift
absolute_shift = secondary_shift + random_coil[atom][resname]
```

Random coil values depend on whether the next residue is Proline (uses `randcoil_pro` vs `randcoil_ala`).

### Feature Column Naming
Features follow a strict naming convention with position suffixes:
- `_i-1`: Previous residue
- `_i`: Current residue
- `_i+1`: Next residue

Examples: `PHI_COS_i-1`, `CHI1_SIN_i`, `S2_i+1`

### Ring Current Ambiguity
For glycine HA atoms, there's HA2/HA3 ambiguity. The `ha23ambigfix()` function resolves this:
- Mode 0: Average HA2 and HA3
- Mode 1: Use HA2 only
- Mode 2: Use HA3 only

### Residue Name Mappings
Non-standard residue names are mapped to standard ones via `EXTERNAL_MAPPINGS`:
```python
EXTERNAL_MAPPINGS = {"HIE":"HIS", "HID":"HIS", "HIP":"HIS",
                     "CAS":"CYS", "CSD":"CYS", "MSE":"MET", "CSO":"CYS"}
```

## Debugging Tips

### Checking Feature Extraction
If predictions seem wrong, verify features are extracted correctly:
```python
from spartap_features import PDB_SPARTAp_DataReader
reader = PDB_SPARTAp_DataReader()
df = reader.df_from_file_3res("test.pdb")
print(df.columns.tolist())  # Check expected columns exist
print(df.isnull().sum())     # Check for excessive NaN
```

### DSSP Failures
DSSP may fail on residues with missing atoms. Check for KeyError/TypeError exceptions in feature extraction output.

### Alignment Issues
If UCBShift-Y returns NaN for all atoms:
1. Check BLAST database is accessible: `BLASTDB` env var
2. Verify mTM-align binary is executable
3. Look for "No sequence in database generates possible alignments" message

### Model Loading Errors
scikit-learn models are version-sensitive. If `joblib.load()` fails:
- Ensure scikit-learn 0.22 is installed
- Models trained with different sklearn versions are incompatible

## Reference Database (refDB)

The `refDB/` directory contains:
- **2386 reference proteins** with known chemical shifts
- `shifts_df/*.csv`: Per-protein shift tables
- `pdbs.tgz`: Compressed PDB structures (auto-extracted on first run)
- BLAST databases for sequence search

Reference proteins are from BMRB with experimentally determined shifts.

## Performance Optimization

### ONNX GPU Acceleration

The ML inference pipeline supports ONNX Runtime for GPU-accelerated predictions. Convert sklearn models to ONNX format:

```bash
pip install onnx skl2onnx onnxruntime-gpu
python scripts/convert_to_onnx.py --models-dir models/
```

GPU inference is used automatically when `.onnx` files exist alongside `.sav` files. Models are cached to avoid repeated loading.

### DIAMOND for Faster Alignment

DIAMOND provides ~100x faster protein alignment than BLAST for UCBShift-Y:

```bash
# Install DIAMOND (download from https://github.com/bbuchfink/diamond/releases)
chmod +x bins/diamond

# Build database (one-time)
./scripts/build_diamond_db.sh
```

DIAMOND is used automatically when `bins/diamond` and `refDB/refDB.dmnd` exist. Falls back to BLAST otherwise.

### Batch Processing with Multiprocessing

For multiple PDB files, use batch mode with parallel workers:

```bash
python CSpred.py batch_list.txt -b -w 8 -o output_dir/
```

The `-w` flag controls the number of parallel workers. Each worker loads its own copy of the models (uses spawn context to avoid memory sharing issues).

### Model Caching

Models are automatically cached on first load via `_MODEL_CACHE`. For batch processing, call `preload_models()` before predictions to load all models upfront.

### Performance Bottlenecks

- **UCBShift-Y alignment (~60%)**: DIAMOND provides major improvement
- **Feature extraction (~25%)**: Parallelized via batch multiprocessing
- **ML inference (~10%)**: ONNX GPU provides modest improvement

See `docs/GPU_OPTIMIZATION.md` for detailed optimization guide including PyTorch model retraining plans.
