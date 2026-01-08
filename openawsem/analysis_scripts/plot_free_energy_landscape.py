import os
import glob
import argparse
import json

import multiprocessing as mp
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm


from MDAnalysis.analysis import rms

def plot_population_density(rmsd_data, filename="population_map.png", use_log_scale=False):
    """
    Plots a 2D density map showing how often each RMSD state was sampled.
    """
    # 1. Create 2D Histogram
    max_A = np.ceil(rmsd_data[:, 0].max() * 10) / 10.0 + 0.1
    max_B = np.ceil(rmsd_data[:, 1].max() * 10) / 10.0 + 0.1
    bins = 100 # Increased bins for better density resolution
    
    counts, x_edges, y_edges = np.histogram2d(
        rmsd_data[:, 0], 
        rmsd_data[:, 1], 
        bins=bins,
        range=[[0, max_A], [0, max_B]]
    )

    # 2. Convert to Probability Density
    # P represents the fraction of total simulation time spent in that bin
    P = counts.T / np.sum(counts) 

    # 3. Prepare coordinates
    X_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    
    plt.figure(figsize=(10, 8))

    # 4. Plotting
    if use_log_scale:
        # We use a small floor to avoid log(0) issues for empty bins
        P_plot = np.log10(P + 1e-6)
        label = r"$\log_{10}(\mathrm{Population\ Fraction})$"
    else:
        P_plot = P
        label = "Population Fraction (Probability)"

    # Use pcolormesh for a clean "heatmap" look, or contourf for smooth gradients
    mesh = plt.pcolormesh(X_centers, Y_centers, P_plot, cmap='magma', shading='auto')
    
    # Add Colorbar
    cbar = plt.colorbar(mesh)
    cbar.set_label(label, fontsize=14)

    # Formatting
    plt.xlabel(r'$\mathrm{RMSD_{Segment\ A}}$ ($\mathrm{\AA}$)', fontsize=14)
    plt.ylabel(r'$\mathrm{RMSD_{Segment\ B}}$ ($\mathrm{\AA}$)', fontsize=14)
    plt.title('Protein Conformational Sampling Density', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename)

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

# def load_and_align_trajectory(trajectory_file, reference_file, selection):
#     """Loads a trajectory and aligns all frames to the reference structure."""
#     if not os.path.exists(trajectory_file):
#         print(f"Warning: File not found: {trajectory_file}. Skipping.")
#         return None

#     # Load the reference and the trajectory
#     ref = mda.Universe(reference_file)
#     mobile = mda.Universe(trajectory_file)

#     # Check for empty universe after loading (e.g., if PDB is malformed)
#     if not mobile.select_atoms("all"):
#         print(f"Error: Could not load atoms from {trajectory_file}. Skipping.")
#         return None

#     # Perform alignment on the selected atoms
#     aligner = rms.RMSD(
#         mobile,
#         ref,
#         select=selection,
#         groupselections=[selection, selection] # Both groups should use the same selection
#     )
#     aligner.run()

#     # The mobile Universe is now implicitly aligned via the results object
#     return mobile, aligner.results.rmsd, ref, mobile

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

def plot_free_energy_landscape(rmsd_data, T, filename="fel_plot.png"):
    """
    Calculates the Free Energy Landscape (FEL) from 2D RMSD data 
    and generates a contour plot.
    
    Args:
        rmsd_data (np.ndarray): N x 2 array of RMSD values.
        T (float): Simulation temperature in Kelvin.
        filename (str): Output filename for the plot.
    """
    R = 8.31446 / 1000.0  # Gas constant in kJ/(mol*K)
    kT = R * T            # kT in kJ/mol

    # 1. Create 2D Histogram (Binning)
    # Automatically determine bin range and use 50x50 bins
    max_A = np.ceil(rmsd_data[:, 0].max() * 10) / 10.0 + 0.1
    max_B = np.ceil(rmsd_data[:, 1].max() * 10) / 10.0 + 0.1
    bins = 50
    
    counts, x_edges, y_edges = np.histogram2d(
        rmsd_data[:, 0],       # RMSD Segment A (X-axis)
        rmsd_data[:, 1],       # RMSD Segment B (Y-axis)
        bins=bins,
        range=[[0, max_A], [0, max_B]]
    )

    # 2. Convert Counts to Probability (P)
    # Transpose P for correct plotting orientation (x vs y)
    P = counts.T / np.sum(counts) 
    
    # 3. Calculate Free Energy (Delta G)
    # G = -kT * ln(P/P_max)
    P_max = P.max()
    
    # Set a small minimum probability to avoid log(0)
    min_P = 1e-10
    P[P < min_P] = min_P 
    
    G = -kT * np.log(P / P_max)

    # 4. Prepare coordinates for plotting (centers of the bins)
    X_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    
    # 5. Plotting
    plt.figure(figsize=(10, 8))

    # Define contour levels (e.g., in steps of 5 kJ/mol)
    G_max_plot = np.ceil(G.max()/5) * 5
    levels = np.arange(0, G_max_plot, 5)

    # Plot filled contours (FEL surface)
    CF = plt.contourf(X_centers, Y_centers, G, levels=levels, cmap='viridis_r', extend='max')

    # Add black contour lines for clarity
    CS = plt.contour(X_centers, Y_centers, G, levels=levels, colors='k', linewidths=0.5)
    
    # Label the contour lines
    plt.clabel(CS, inline=1, fontsize=10, fmt='%.1f')

    # Add a color bar
    cbar = plt.colorbar(CF, ticks=levels)
    cbar.set_label(r'$\Delta G$ $(\mathrm{kJ/mol})$', fontsize=14)

    # Set axis labels with LaTeX formatting
    plt.xlabel(r'$\mathrm{RMSD_{Segment\ A}}$ ($\mathrm{\AA}$)', fontsize=14)
    plt.ylabel(r'$\mathrm{RMSD_{Segment\ B}}$ ($\mathrm{\AA}$)', fontsize=14)
    
    # Set title
    plt.title(r'Free Energy Landscape at $T = ' + f'{T}' + r'\ \mathrm{K}$', fontsize=16)
    
    plt.xlim(X_centers.min(), X_centers.max())
    plt.ylim(Y_centers.min(), Y_centers.max())

    plt.tight_layout()
    plt.savefig(filename)


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
    parser.add_argument('-tm', '--temp_map',
                        required=True,
                        help="JSON file mapping state IDs to temperatures")
    parser.add_argument("-ct",
                        "--convert_temp",
                        type=bool, default=False,
                        help="Set to True to convert temperature units from K to C by subtracting 273")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=mp.cpu_count())
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.input_dir, args.input_pattern)))
    # load temp map
    temp_map, temp_unit = load_temp_map(args)


    # Dictionary to hold all results
    # all_rmsd_results = {}

    print(f"Starting analysis with Reference: {args.ref}...")

    # for file_path in files:
    #     file_tag = os.path.basename(file_path).replace(".pdb", "")
    #     print(f"\n--- Analyzing: {file_tag} ---")

    #     # Step 1: Load and Align
    #     rmsd_data = calculate_segment_rmsd_for_fel(file_path, args.ref)
    #     all_rmsd_results[file_tag] = rmsd_data


    with mp.Pool(processes=args.jobs) as pool:
        task_args = [(f, args.ref, args.selection) for f in files]
        results_list = pool.starmap(calculate_segment_rmsd_for_fel, task_args)

    # Filter out errors and collect data
    valid_results = [res for res in results_list if isinstance(res, np.ndarray)]
    errors = [res for res in results_list if isinstance(res, str)]
    
    for err in errors:
        print(err)

    results_dict = dict(zip([f.split('.')[0] for f in files], results_list))

    print("\n--- Analysis Complete ---")
    print("\n--- Plotting ---")

    for file_tag, data in results_dict.items():
        plot_free_energy_landscape(data,
                                   T=temp_map[int(file_tag.split('_')[-2])],
                                   filename=f"{file_tag}_FEL.png")
        
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

    # Step 3: Determine Temperature
    # For a combined plot, use the average temperature or a specific reference temp
    avg_temp = np.mean(list(temp_map.values()))
    
    plot_free_energy_landscape(
        final_data, 
        T=avg_temp, 
        filename="Combined_Protein_FEL.png"
    )
    
    plot_population_density(
            final_data, 
            T=avg_temp, 
            filename="Combined_Protein_Polulation_density.png"
        )

    print("Done. Saved as Combined_Protein_FEL and population density figures.png")

if __name__ == "__main__":
    main()
