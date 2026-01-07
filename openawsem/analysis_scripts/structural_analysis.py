import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
import json
import multiprocessing as mp
from MDAnalysis.analysis import rms, distances
from MDAnalysis.analysis.dssp import DSSP

def process_state_file(f, ref_file, selection, temp_map):
    """
    Processes a single state file and returns structural metrics.
    """
    # Extract state number from filename (assumes 'state_X_AA.pdb' format)
    try:
        filename = os.path.basename(f)
        state_id = filename.split('_')[1] 
        temperature = temp_map.get(state_id) or temp_map.get(int(state_id))
        
        if temperature is None:
            return f"Error: State {state_id} not found in JSON map."
    except Exception:
        return f"Error: Could not parse state ID from {f}"

    try:
        ref = mda.Universe(ref_file)
        u = mda.Universe(f)
        
        # Selection for alignment
        aligner = rms.RMSD(u, ref, select=selection).run()
        rmsd_vals = aligner.results.rmsd[:, 2]
        
        protein = u.select_atoms('protein')
        
        # Radius of Gyration
        rg_vals = [protein.radius_of_gyration() for ts in u.trajectory]
        
        # Q3 Secondary Structure
        q3_avg = 0
        try:
            # Trim possibly incomplete termini
            first_res = protein.residues[0].resid
            last_res = protein.residues[-1].resid
            residue_range_sel = f"protein and not (resnum {first_res} or" \
                f" resname {last_res}) and (name N or name CA or name O)"
            trimmed_backbone = u.select_atoms(residue_range_sel)

            # Check trimming worked:
            n_residues = len(trimmed_backbone.residues)
            n_n = len(trimmed_backbone.select_atoms("name N"))
            n_ca = len(trimmed_backbone.select_atoms("name CA"))

            if n_n != n_ca or n_n != n_residues:
                raise ValueError("Termini trimming failed"
                                 f"Current counts: N={n_n}, CA={n_ca}, "
                                 f"Total residue number {n_residues}")

            dssp = DSSP(trimmed_backbone).run()
            q3_per_frame = [np.sum(np.isin(frame, ['H', 'E', 'G'])) / n_residues for frame in dssp.results.dssp]
            q3_avg = np.mean(q3_per_frame)
        except:
            q3_avg = np.nan

        # End-to-End Distance
        e2e_avg = 0
        try:
            n_term = protein.residues[0].atoms.select_atoms("name CA")
            c_term = protein.residues[-1].atoms.select_atoms("name CA")
            e2e_vals = [distances.distance_array(n_term.positions, c_term.positions)[0][0] for ts in u.trajectory]
            e2e_avg = np.mean(e2e_vals)
        except:
            e2e_avg = np.nan

        return {
            'Temperature': float(temperature),
            'State': state_id,
            'Avg_RMSD': np.mean(rmsd_vals),
            'Avg_Rg': np.mean(rg_vals),
            'Avg_Q3': q3_avg,
            'Avg_E2E': e2e_avg
        }

    except Exception as e:
        return f"Error in {filename}: {str(e)}"

def plot_metrics_vs_temp(df, base_name):
    """Generates plots for structural metrics vs Temperature."""
    
    # 1. Force numeric conversion for the columns we want to plot
    # This ensures strings like '1.0' or '300' become actual floats
    cols_to_fix = ['Temperature', 'Avg_Rg', 'Avg_Q3', 'Avg_E2E', 'Avg_RMSD']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Use numeric_only=True in the mean() call
    # This prevents pandas from trying to average the 'State' strings
    summary = df.groupby('Temperature').mean(numeric_only=True).sort_index().reset_index()
    
    metrics = [('Avg_Rg', 'Radius of Gyration (Å)'), 
               ('Avg_Q3', 'Q3 Content'), 
               ('Avg_E2E', 'End-to-End Distance (Å)'),
               ('Avg RMSD', 'Root-mean-square-fluctuation (Å)')]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    
    for i, (col, label) in enumerate(metrics):
        axes[i].plot(summary['Temperature'], summary[col], marker='o', linestyle='-', color='teal')
        axes[i].set_xlabel('Temperature (K)')
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} vs T')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_name}_temp_trends.png", dpi=300)
    print(f"Trend plots saved as {base_name}_temp_trends.png")

def main():
    parser = argparse.ArgumentParser(description="REMD Data Analysis: Metrics vs Temperature")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--json_map", required=True, help="JSON file mapping state IDs to Temperatures")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref", default="native_structure.pdb")
    parser.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    # Load Temperature Map
    with open(args.json_map, 'r') as f:
        temp_map = json.load(f)

    files = sorted(glob.glob(os.path.join(args.input_dir, "state_*_AA.pdb")))
    selection = "protein and name CA"
    
    if not files:
        print("No files found matching the pattern state_*_AA.pdb")
        return

    print(f"Processing {len(files)} files on {args.jobs} cores...")
    
    with mp.Pool(processes=args.jobs) as pool:
        task_args = [(f, args.ref, selection, temp_map) for f in files]
        results_list = pool.starmap(process_state_file, task_args)

    # Filter out errors and collect data
    valid_results = [res for res in results_list if isinstance(res, dict)]
    errors = [res for res in results_list if isinstance(res, str)]
    
    for err in errors:
        print(err)

    if valid_results:
        df = pd.DataFrame(valid_results)
        df.to_csv(f"{args.name}_temperature_summary.csv", index=False)
        
        # Generate the plots
        plot_metrics_vs_temp(df, args.name)
        print("Analysis complete.")

if __name__ == "__main__":
    main()
