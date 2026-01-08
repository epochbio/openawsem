import os
import glob
import argparse
import json

import multiprocessing as mp
import MDAnalysis as mda
import numpy as np

import matplotlib.pyplot as plt

from MDAnalysis.analysis import rms


def load_temp_map(args):
    """Load temp map from JSON, and convert to Celsius if specified"""
    with open(args.temp_map, 'r') as f:
        temp_map = json.load(f)

    temp_unit = 'K' # default unit is Kelvin
    if args.convert_temp:
        # Remove 273 to convert to celsius.
        for id, temp in temp_map.items():
            temp_map[id] = temp - 273
        temp_unit = 'C'

    return temp_map, temp_unit

def calculate_segment_rmsd_for_fel(trajectory_file, reference_file,
                                   main_selection='backbone'):
    """
    Calculates the RMSD for two overlapping segments (first 3/5ths and last 3/5ths)
    of a protein trajectory, after aligning the entire structure.

    Args:
        trajectory_file (str): Path to the trajectory file (e.g., .xtc, .dcd).
        reference_file (str): Path to the reference structure file (e.g., .pdb).
        main_selection (str): Optional. Selection string for alignment e.g 'backbone'.

    Returns:
        np.ndarray or None: A N x 2 array where N is the number of frames.
                            Column 0 is RMSD(Segment A), Column 1 is RMSD(Segment B).
                            Returns None on error.
    """
    if not os.path.exists(trajectory_file):
        print(f"Warning: File not found: {trajectory_file}. Skipping.")
        return None

    # Load the reference and the trajectory
    ref = mda.Universe(reference_file)
    mobile = mda.Universe(trajectory_file)

    # 1. Check for atom count and calculate segment boundaries
    n_residues = len(ref.residues)
    if n_residues == 0:
        print(f"Error: No residues found in the reference structure.")
        return None

    # Calculate the split points
    # The selection is based on residue indices (resnum), which typically start at 1.
    split_point_A = int(np.ceil(0.6 * n_residues)) # 3/5ths point for Segment A end
    split_point_B = n_residues - int(np.floor(0.6 * n_residues)) + 1 # 3/5ths point for Segment B start

    # 2. Define the selection strings
    # Segment A: First 3/5ths (Residues 1 to split_point_A)
    # Segment B: Last 3/5ths (Residues split_point_B to n_residues)

    # Use the whole protein's backbone for global alignment
    selection_A = f"resid 1:{split_point_A} and {main_selection}"
    selection_B = f"resid {split_point_B}:{n_residues} and {main_selection}"

    print(f"Total residues: {n_residues}")
    print(f"Segment A selection: {selection_A} (Approx residues 1 to {split_point_A})")
    print(f"Segment B selection: {selection_B} (Approx residues {split_point_B} to {n_residues})")
    print(f"Alignment selection: {main_selection}")

    # 3. Perform GLOBAL alignment on the entire trajectory
    # Align the 'mobile' trajectory using the entire structure selection to the reference
    aligner = rms.RMSD(
        mobile,
        ref,
        select=main_selection,
        groupselections=[main_selection, main_selection]
    )
    aligner.run()

    # The trajectory (mobile) is now aligned implicitly by the alignment data stored in aligner.results.
    # The actual coordinates are NOT changed in the mobile Universe, 
    # but the MDAnalysis RMSD classes know how to apply the transforms.

    # 4. Calculate Segment RMSDs using the ALIGNED trajectory
    # The RMSD is calculated using the ALIGNED coordinates relative to the REFERENCE.
    rmsd_A = rms.RMSD(
        mobile,
        ref,
        select=selection_A,
        groupselections=[main_selection, selection_A] # Align to ALL, then calculate RMSD on A
    ).run().results.rmsd[:, 2] # Extract the RMSD column (index 2)

    rmsd_B = rms.RMSD(
        mobile,
        ref,
        select=selection_B,
        groupselections=[main_selection, selection_B] # Align to ALL, then calculate RMSD on B
    ).run().results.rmsd[:, 2] # Extract the RMSD column (index 2)

    # Combine the results into a single N x 2 array
    # N x 3 (Frame, Time, RMSD) -> N x 1 (RMSD)
    rmsd_data = np.column_stack((rmsd_A, rmsd_B))

    return rmsd_data

def plot_combined_population(rmsd_data, filename="combined_population.png"):
    """
    Plots a 2D Heatmap of the sampled RMSD space across all combined states.
    """
    # 1. Setup the grid
    # We look at the max RMSD across the whole combined dataset to set limits
    max_A = np.max(rmsd_data[:, 0]) + 0.5
    max_B = np.max(rmsd_data[:, 1]) + 0.5
    bins = 100 
    
    # 2. Calculate Histogram
    counts, x_edges, y_edges = np.histogram2d(
        rmsd_data[:, 0], 
        rmsd_data[:, 1], 
        bins=bins,
        range=[[0, max_A], [0, max_B]]
    )

    # 3. Normalize to Population Fraction
    # Total frames across all simulations combined
    P = counts.T / np.sum(counts) 

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    
    # pcolormesh is better for 'frequency' maps than contourf
    X, Y = np.meshgrid(x_edges, y_edges)
    mesh = plt.pcolormesh(X, Y, P, cmap='CMRmap_r', shading='auto')
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('Population Fraction', fontsize=12)

    plt.xlabel(r'$\mathrm{RMSD_{Segment\ A}}$ ($\mathrm{\AA}$)', fontsize=14)
    plt.ylabel(r'$\mathrm{RMSD_{Segment\ B}}$ ($\mathrm{\AA}$)', fontsize=14)
    plt.title('Combined Population Density Map', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved combined plot to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generation of Free Energy" \
    "Landscapes using two sections of a provided structure.")
    parser.add_argument("-i", "--input_dir",
                        required=True,
                        help="Path to directory containing all-atom state PDBs")
    parser.add_argument("-ip", "--input_pattern",
                        default="state_*_AA.pdb",
                        help="Filename pattern to search for trajectories of." \
                        " Default: 'state_*_AA.pdb'")
    parser.add_argument("-r", "--ref",
                        required=True,
                        help="Path to all-atom PDB reference to align to")
    parser.add_argument('-s', '--selection',
                        default='protein and name CA',
                        help="Biopython selection string for alignment and " \
                        "analsis (e.g 'protein' or 'protein and backbone'.). " \
                        "Default is 'protein and name CA'.")
    parser.add_argument('-o', '--output',
                        default="combined",
                        help="Output file name. Format will be " \
                        "X_population_density.png, default is" \
                        " combined_population_density.png")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=mp.cpu_count())
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.input_dir, args.input_pattern)))

    print(f"Starting analysis with Reference: {args.ref}...")

    # rename output file
    outfile = f'{args.output}_population_density.png'
    

    with mp.Pool(processes=args.jobs) as pool:
        task_args = [(f, args.ref, args.selection) for f in files]
        results_list = pool.starmap(calculate_segment_rmsd_for_fel, task_args)

    # Filter out errors and collect data
    errors = [res for res in results_list if isinstance(res, str)]
    
    for err in errors:
        print(err)


    print("\n--- Analysis Complete ---")
    print("\n--- Plotting ---")
        
    # Step 2: Aggregate Data
    combined_rmsd_data = []
    
    for i, res in enumerate(results_list):
        if isinstance(res, np.ndarray):
            combined_rmsd_data.append(res)
        else:
            print(f"Warning: Result for {files[i]} was invalid.")

    if not combined_rmsd_data:
        print("No valid data found to plot.")
        return

    # Stack all N x 2 arrays into one large (Total_N) x 2 array
    final_data = np.vstack(combined_rmsd_data)

    print(f"\n--- Analysis Complete. Combined Data Points: {len(final_data)} ---")
    print("\n--- Generating Combined Plot ---")

    
    plot_combined_population(
        final_data, 
        filename=outfile
    )

    print(f"Done. {outfile}")

if __name__ == "__main__":
    main()
