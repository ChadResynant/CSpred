#!/bin/bash
# Build DIAMOND database from refDB FASTA for faster sequence search
#
# Usage: ./scripts/build_diamond_db.sh
#
# Requirements: DIAMOND must be installed
#   Download from: https://github.com/bbuchfink/diamond/releases

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DIAMOND_EXE="$REPO_DIR/bins/diamond"
REFDB_FASTA="$REPO_DIR/refDB/refDB.fasta"
REFDB_DMND="$REPO_DIR/refDB/refDB.dmnd"

# Check for DIAMOND
if [ ! -x "$DIAMOND_EXE" ]; then
    echo "DIAMOND not found at $DIAMOND_EXE"
    echo ""
    echo "To install DIAMOND:"
    echo "  1. Download from https://github.com/bbuchfink/diamond/releases"
    echo "  2. Extract and copy to $REPO_DIR/bins/diamond"
    echo "  3. Make executable: chmod +x $REPO_DIR/bins/diamond"
    exit 1
fi

# Check for input FASTA
if [ ! -f "$REFDB_FASTA" ]; then
    echo "Reference FASTA not found at $REFDB_FASTA"
    exit 1
fi

# Build database
echo "Building DIAMOND database from $REFDB_FASTA..."
"$DIAMOND_EXE" makedb --in "$REFDB_FASTA" -d "${REFDB_DMND%.dmnd}"

if [ -f "$REFDB_DMND" ]; then
    echo "Success! DIAMOND database created at $REFDB_DMND"
    echo ""
    echo "UCBShift-Y will now use DIAMOND automatically (100x faster than BLAST)"
else
    echo "Error: Failed to create DIAMOND database"
    exit 1
fi
