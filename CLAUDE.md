# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UCBShift is a protein chemical shift predictor for backbone and side chain atoms in aqueous solution. It combines two prediction modules:

- **UCBShift-X**: Machine learning module using ensemble decision trees
- **UCBShift-Y**: Transfer prediction module using sequence (BLAST) and structure (mTM-align) alignments

Published accuracy (RMSE): H: 0.38, Hα: 0.22, C': 1.31, Cα: 0.97, Cβ: 1.29, N: 2.16 ppm

## Quick Start

```bash
# Basic prediction
python CSpred.py your_protein.pdb

# With pH specification
python CSpred.py protein.pdb --pH 7

# UCBShift-Y only (transfer prediction)
python CSpred.py protein.pdb --shifty_only

# UCBShift-X only (machine learning)
python CSpred.py protein.pdb --shiftx_only

# Batch mode
python CSpred.py input_list.txt --batch -o output_dir/
```

## Setup Requirements

**Models**: Download from https://doi.org/10.5281/zenodo.15375968 and extract to `models/` (141 .sav files)

**External programs** (bundled in `bins/`):
- BLAST 2.9.0+ (`bins/ncbi-blast-2.9.0+/`)
- mTM-align (`bins/mTM-align/`)
- DSSP (`bins/mkdssp`)
- reduce (must be in PATH for hydrogenation)

**Python dependencies** (scikit-learn==0.22 is critical for model compatibility):
```bash
pip install -r requirements.txt
```

## Architecture

### Entry Points

| File | Purpose |
|------|---------|
| `CSpred.py` | Main entry - combines UCBShift-X and UCBShift-Y |
| `ucbshifty.py` | UCBShift-Y standalone (sequence/structure alignment) |

### Prediction Pipeline

1. **Feature extraction** (`spartap_features.py`): Extracts ~200+ features from PDB including dihedral angles, hydrogen bonds, ring currents, BLOSUM62 scores, HSE, DSSP secondary structure
2. **Data preprocessing** (`data_prep_functions.py`): Feature normalization, polynomial transformations, random coil subtraction
3. **UCBShift-Y** (`ucbshifty.py`): BLAST → mTM-align → shift transfer from reference DB
4. **UCBShift-X** (`CSpred.py`): R0 model → R1 model (stacked) → R2 model (when Y available)

### Model Structure

Per atom type (47 atoms total), three model tiers:
- `{ATOM}_R0.sav`: Base prediction model
- `{ATOM}_R1.sav`: Stacked model using R0 output
- `{ATOM}_R2.sav`: Combined model using R1 + UCBShift-Y predictions

### Key Data Structures

**Feature columns** (defined in `data_prep_functions.py`):
- `col_phipsi`: Backbone dihedral angles (sin/cos for i-1, i, i+1)
- `col_chi`: Side chain dihedral angles (CHI1-5, including alternate conformations)
- `col_hbond`: Hydrogen bond features (distance, angles)
- `hse_cols`: Half-sphere exposure
- `dssp_cols`: Secondary structure, ASA, hydrogen bond energies
- `ring_cols`: Ring current contributions (`{ATOM}_RC`)
- `rcoil_cols`: Random coil shifts (`RCOIL_{ATOM}`)

**Reference database** (`refDB/`):
- `pdbs/`: Single-chain PDB structures
- `shifts_df/`: CSV files with experimental shifts per structure
- `refDB.blastdb.*`: BLAST database for sequence alignment

## Training Pipeline

Located in `train_model/`:

1. `download_pdbs.py` - Download PDB files, add hydrogens
2. `protonation.py` - pH-dependent protonation via PDB2PQR
3. `build_df.py` - Extract features to CSV
4. `make_Y_preds.py` - Generate UCBShift-Y predictions for training
5. `train.py` - Train models for all atom types
6. `evaluate.py` - Evaluate on test set

## Important Constants

**Atom types** (`toolbox.ATOMS`): 47 atoms including backbone (C, CA, CB, N, H, HA) and side chain carbons/protons/nitrogens

**Residue mappings** (`ucbshifty.EXTERNAL_MAPPINGS`): HIE/HID/HIP→HIS, MSE→MET, CSO/CSD/CAS→CYS

**Random coil shifts**: Wishart et al. J-Bio NMR 1995, with Pro-following corrections (`randcoil_ala`, `randcoil_pro`)

## BioPython Compatibility

The codebase handles BioPython API changes:
- `blosum62` import from either `Bio.SubsMat.MatrixInfo` (old) or `Bio.Align.substitution_matrices` (new)
- `three_to_one` function compatibility layer in `spartap_features.py`
