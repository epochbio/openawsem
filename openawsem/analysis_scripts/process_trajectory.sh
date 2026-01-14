#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Argument Parsing ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <protein_name> <ref_temp> <input_directory>"
    echo "Example: $0 PRO1282_no_NC 62.6 input_directory"
    exit 1
fi

PROTEIN=$1
REF_TEMP=$2
TARGET_DIR=$3

# --- Environment Check ---
# Ensure your environment variables are set
OA=${OPENAWSEM_LOCATION:-$OA}

# --- Execution ---

echo "Moving into directory: $TARGET_DIR"
cd "$TARGET_DIR"

echo "Setting up folder structure..."
# Create folders and move existing files into 01_replica_exchange
mkdir -p 01_replica_exchange 02_heat_capacity 03_native_contact 04_output
# Move everything except the newly created directories into 01_replica_exchange
find . -maxdepth 1 -not -name '0*' -not -name '.' -exec mv {} 01_replica_exchange/ \;

cd 01_replica_exchange/
mkdir -p output
# Move any existing output files/folders into the output directory
find . -maxdepth 1 -name "output*" -exec mv {} output/ \;

echo "Running OpenAWSEM replica exchange processing..."
# Ensure the conda environment is active (Note: scripts usually need 'source' for conda)
# eval "$(conda shell.bash hook)"
# conda activate openawsem
python "$OA/scripts/process_remd.py"

echo "Performing structural analysis..."
cp crystal_structure.fasta ../03_native_contact/
cp crystal_structure-openmmawsem.pdb ../03_native_contact/

cd ../03_native_contact/
# Create symlink to trajectories
ln -sf ../01_replica_exchange/output/ traj

# Convert trajectories and generate references
python "$OA/analysis_scripts/dcd_to_aa_pdb.py" traj \
    -f crystal_structure.fasta \
    -r crystal_structure-openmmawsem.pdb \
    -dp "state*dcd"

python "$OA/analysis_scripts/generate_AA_reference.py" \
    crystal_structure-openmmawsem.pdb \
    -f crystal_structure.fasta

echo "Calculating structural metrics and free energy landscape..."
# General structural metrics
python "$OA/analysis_scripts/structural_analysis.py" \
    --input_dir traj \
    --temp_map ../../../temp_map.json \
    --name "$PROTEIN" \
    --ref crystal_structure-openmmawsem_reference_AA.pdb \
    -ct True -rt "$REF_TEMP"

# Population landscape
python "$OA/analysis_scripts/plot_free_energy_landscape.py" \
    -i traj \
    -r crystal_structure-openmmawsem_reference_AA.pdb

# Move images to output folder
cd ..
cp 01_replica_exchange/output/*.png 04_output/
cp 03_native_contact/*.png 04_output/
cp 03_native_contact/*.json 04_output/
cp 03_native_contact/*.csv 04_output/

echo "Process complete for $PROTEIN."
