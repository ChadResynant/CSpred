#!/usr/bin/env python3
"""
Database Expansion Script for UCBShift/CSpred

This script expands the reference database (refDB) using two approaches:
1. New experimental data from BMRB (Biological Magnetic Resonance Bank)
2. QM-calculated chemical shifts for underrepresented conformations

Usage:
    python scripts/expand_database.py --mode bmrb     # Download new BMRB entries
    python scripts/expand_database.py --mode qm       # Generate QM training data
    python scripts/expand_database.py --mode analyze  # Analyze coverage gaps

Requirements:
    - pynmrstar (pip install pynmrstar)
    - qm_nmr from ~/repos/quantumchemistry
    - Biopython
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).parent.parent
REFDB_PATH = SCRIPT_PATH / "refDB"
SHIFTS_DF_PATH = REFDB_PATH / "shifts_df"
QM_NMR_PATH = Path.home() / "repos" / "quantumchemistry"


def get_existing_bmrb_ids() -> set:
    """Get set of BMRB IDs already in refDB."""
    bmrb_ids = set()
    for f in SHIFTS_DF_PATH.glob("*.csv"):
        # Format: {BMRB_ID}.{PDB_ID}{CHAIN}.csv
        bmrb_id = f.stem.split('.')[0]
        try:
            bmrb_ids.add(int(bmrb_id))
        except ValueError:
            pass
    return bmrb_ids


def analyze_coverage() -> Dict:
    """Analyze current database coverage."""
    print("Analyzing refDB coverage...")

    existing_ids = get_existing_bmrb_ids()
    print(f"  Current entries: {len(existing_ids)}")
    print(f"  BMRB ID range: {min(existing_ids)} - {max(existing_ids)}")

    # Analyze residue type coverage
    all_residues = []
    all_sequences = []

    for f in list(SHIFTS_DF_PATH.glob("*.csv"))[:500]:
        try:
            df = pd.read_csv(f, usecols=['RESNAME'])
            residues = df['RESNAME'].dropna().tolist()
            all_residues.extend(residues)
            all_sequences.append(len(residues))
        except Exception:
            pass

    residue_counts = Counter(all_residues)

    print(f"\n  Total residues sampled: {len(all_residues)}")
    print(f"  Average protein length: {np.mean(all_sequences):.0f} residues")
    print(f"\n  Residue distribution:")
    for res, count in residue_counts.most_common(20):
        print(f"    {res}: {count:5d} ({100*count/len(all_residues):5.1f}%)")

    # Identify underrepresented residues
    expected_freq = 1/20  # ~5% for uniform distribution
    underrepresented = []
    for res in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
        actual_freq = residue_counts.get(res, 0) / len(all_residues)
        if actual_freq < expected_freq * 0.5:  # Less than half expected
            underrepresented.append((res, actual_freq, expected_freq))

    if underrepresented:
        print(f"\n  Underrepresented residues (< 2.5%):")
        for res, actual, expected in underrepresented:
            print(f"    {res}: {100*actual:.1f}% (expected ~{100*expected:.1f}%)")

    return {
        'existing_ids': existing_ids,
        'residue_counts': residue_counts,
        'underrepresented': underrepresented,
        'total_residues': len(all_residues)
    }


def fetch_new_bmrb_entries(max_id: int = None, limit: int = 100) -> List[Dict]:
    """
    Fetch new BMRB entries that aren't in the current database.

    Requires: pip install pynmrstar requests
    """
    try:
        import pynmrstar
        import requests
    except ImportError:
        print("Error: Install required packages: pip install pynmrstar requests")
        return []

    existing_ids = get_existing_bmrb_ids()
    max_existing = max(existing_ids) if existing_ids else 0

    if max_id is None:
        max_id = max_existing + 5000  # Search next 5000 IDs

    print(f"Searching BMRB for entries {max_existing + 1} to {max_id}...")

    new_entries = []

    # BMRB REST API endpoint
    api_base = "https://api.bmrb.io/v2"

    for bmrb_id in range(max_existing + 1, max_id):
        if bmrb_id in existing_ids:
            continue

        try:
            # Check if entry exists and has assigned chemical shifts
            response = requests.get(
                f"{api_base}/entry/{bmrb_id}",
                params={'format': 'json'},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Check for protein chemical shifts
                if 'assigned_chemical_shifts' in str(data).lower():
                    # Try to find associated PDB
                    pdb_ids = []
                    if 'related_entries' in data:
                        for entry in data.get('related_entries', []):
                            if entry.get('database', '').upper() == 'PDB':
                                pdb_ids.append(entry.get('id'))

                    if pdb_ids:
                        new_entries.append({
                            'bmrb_id': bmrb_id,
                            'pdb_ids': pdb_ids,
                            'title': data.get('title', 'Unknown')
                        })
                        print(f"  Found: BMRB {bmrb_id} -> PDB {pdb_ids}")

                        if len(new_entries) >= limit:
                            break

        except Exception as e:
            pass  # Skip entries that fail

    print(f"\nFound {len(new_entries)} new entries with PDB structures")
    return new_entries


def generate_qm_peptide_library(
    residues: List[str] = None,
    conformations: List[Tuple[float, float]] = None,
    output_dir: Path = None
) -> List[Path]:
    """
    Generate QM chemical shifts for peptide fragments covering
    underrepresented regions of conformational space.

    Uses the qm_nmr package from ~/repos/quantumchemistry

    Args:
        residues: List of residue types to calculate (default: underrepresented ones)
        conformations: List of (phi, psi) angles to sample
        output_dir: Directory for output files

    Returns:
        List of paths to generated shift files
    """
    if output_dir is None:
        output_dir = SCRIPT_PATH / "qm_shifts"
    output_dir.mkdir(exist_ok=True)

    # Default: sample underrepresented residues
    if residues is None:
        residues = ['CYS', 'MET', 'TRP', 'HIS']  # Often underrepresented

    # Default: sample key Ramachandran regions
    if conformations is None:
        conformations = [
            # Alpha helix
            (-60, -45),
            # Beta sheet
            (-120, 120),
            (-140, 135),
            # Left-handed helix (for Gly)
            (60, 45),
            # PPII helix
            (-75, 145),
            # Type I turn
            (-60, -30),
            # Type II turn
            (-60, 120),
        ]

    print(f"Generating QM shifts for {len(residues)} residues x {len(conformations)} conformations")
    print(f"Output directory: {output_dir}")

    # Check if qm_nmr is available
    if not QM_NMR_PATH.exists():
        print(f"Error: qm_nmr not found at {QM_NMR_PATH}")
        print("Please ensure ~/repos/quantumchemistry exists")
        return []

    sys.path.insert(0, str(QM_NMR_PATH))

    try:
        from qm_nmr.backends.pyscf_backend import PySCFBackend
        from qm_nmr.calculators import ChemicalShiftCalculator
        from qm_nmr.fragmentation import generate_tripeptide
    except ImportError as e:
        print(f"Error importing qm_nmr: {e}")
        print("Install with: pip install -e ~/repos/quantumchemistry")
        return []

    generated_files = []

    # Use PySCF for fast screening (can switch to ORCA for production)
    backend = PySCFBackend(method='B3LYP', basis='6-31G*')
    calculator = ChemicalShiftCalculator(backend)

    for res in residues:
        for phi, psi in conformations:
            print(f"\n  Calculating {res} at phi={phi}, psi={psi}...")

            try:
                # Generate ACE-X-NME tripeptide fragment
                pdb_file = output_dir / f"{res}_{phi}_{psi}.pdb"

                # This would need implementation in qm_nmr or here
                # generate_tripeptide(res, phi, psi, output_path=pdb_file)

                # Calculate shifts
                # shifts = calculator.calculate(pdb_file, output_dir=output_dir)

                # Save to CSV format compatible with refDB
                # ...

                print(f"    Placeholder: Would calculate shifts for {pdb_file}")

            except Exception as e:
                print(f"    Error: {e}")

    return generated_files


def rebuild_blast_database():
    """Rebuild BLAST and DIAMOND databases after adding new entries."""
    print("\nRebuilding sequence databases...")

    # Collect all sequences from refDB
    fasta_path = REFDB_PATH / "refDB.fasta"

    sequences = []
    for f in SHIFTS_DF_PATH.glob("*.csv"):
        try:
            df = pd.read_csv(f, usecols=['RESNAME'])
            # Convert 3-letter codes to sequence
            seq = ''.join([
                residue_3to1.get(r, 'X')
                for r in df['RESNAME'].dropna()
            ])
            if len(seq) >= 20:  # Minimum length
                entry_id = f.stem
                sequences.append((entry_id, seq))
        except Exception:
            pass

    print(f"  Writing {len(sequences)} sequences to FASTA...")

    with open(fasta_path, 'w') as f:
        for entry_id, seq in sequences:
            f.write(f">{entry_id}\n{seq}\n")

    # Rebuild DIAMOND database
    diamond_exe = SCRIPT_PATH / "bins" / "diamond"
    if diamond_exe.exists():
        import subprocess
        print("  Building DIAMOND database...")
        subprocess.run([
            str(diamond_exe), 'makedb',
            '--in', str(fasta_path),
            '-d', str(REFDB_PATH / "refDB")
        ], check=True)

    # Rebuild BLAST database
    makeblastdb = SCRIPT_PATH / "bins" / "ncbi-blast-2.9.0+" / "bin" / "makeblastdb"
    if makeblastdb.exists():
        import subprocess
        print("  Building BLAST database...")
        subprocess.run([
            str(makeblastdb),
            '-in', str(fasta_path),
            '-dbtype', 'prot',
            '-out', str(REFDB_PATH / "refDB.blastdb")
        ], check=True)

    print("  Done!")


# 3-letter to 1-letter amino acid code mapping
residue_3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def main():
    parser = argparse.ArgumentParser(description='Expand UCBShift reference database')
    parser.add_argument('--mode', choices=['analyze', 'bmrb', 'qm', 'rebuild'],
                        default='analyze',
                        help='Operation mode')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum new entries to fetch (bmrb mode)')
    parser.add_argument('--residues', nargs='+',
                        help='Residues to calculate (qm mode)')
    args = parser.parse_args()

    if args.mode == 'analyze':
        coverage = analyze_coverage()

        print("\n" + "="*60)
        print("EXPANSION RECOMMENDATIONS")
        print("="*60)
        print(f"""
1. BMRB Updates:
   - Current max BMRB ID: {max(coverage['existing_ids'])}
   - Run: python scripts/expand_database.py --mode bmrb
   - This will fetch new experimental entries

2. QM Augmentation:
   - Underrepresented residues: {[r[0] for r in coverage['underrepresented']]}
   - Run: python scripts/expand_database.py --mode qm
   - Uses ~/repos/quantumchemistry for DFT calculations

3. After expansion:
   - Run: python scripts/expand_database.py --mode rebuild
   - This rebuilds BLAST/DIAMOND databases
""")

    elif args.mode == 'bmrb':
        new_entries = fetch_new_bmrb_entries(limit=args.limit)
        if new_entries:
            print("\nTo download these entries, use train_model/download_pdbs.py")

    elif args.mode == 'qm':
        residues = args.residues or ['CYS', 'MET', 'TRP', 'HIS']
        generate_qm_peptide_library(residues=residues)

    elif args.mode == 'rebuild':
        rebuild_blast_database()


if __name__ == '__main__':
    main()
