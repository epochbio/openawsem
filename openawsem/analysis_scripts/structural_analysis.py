import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import argparse
import glob
import json
import multiprocessing as mp
from MDAnalysis.analysis import rms, distances
from MDAnalysis.analysis.dssp import DSSP

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

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
            print('Attempting Q3 calculations')
            # Trim possibly incomplete termini
            first_res = protein.residues[0].resid
            last_res = protein.residues[-1].resid
            residue_range_sel = f"protein and not (resnum {first_res} or " \
                f"resnum {last_res}) and (name N or name " \
                    " CA or name C or name O)"
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
        except Exception as e:
            print(f'Q3 calculation failed with error: {e}')
            import traceback
            # Print traceback for error
            traceback.print_exc()
            q3_avg = np.nan

        # RMSF (Root Mean Square Fluctuation)
        # RMSF measures the average distance a residue is from a reference position.
        # High RMSF indicates high flexibility, often observed in unfolded or loop regions.
        R = rms.RMSF(protein, select=selection).run()
        # The output is per-atom, we usually look at per-residue averages or C-alpha only.
        rmsf_data = R.results.rmsf

        # End-to-End Distance
        e2e_avg = 0
        try:
            n_term = protein.residues[0].atoms.select_atoms("name CA")
            c_term = protein.residues[-1].atoms.select_atoms("name CA")
            e2e_vals = [distances.distance_array(n_term.positions, c_term.positions)[0][0] for ts in u.trajectory]
            e2e_avg = np.mean(e2e_vals)
        except:
            e2e_avg = np.nan

        avg_dict = {
            'Temperature': float(temperature),
            'State': state_id,
            'Avg_RMSD': np.mean(rmsd_vals),
            'Avg_Rg': np.mean(rg_vals),
            'Avg_Q3': q3_avg,
            'Avg_E2E': e2e_avg
            }
        full_dict = {
            'Temperature': float(temperature),
            'State': state_id,
            'RMSD': rmsd_vals,
            'Rg': rg_vals,
            'Q3': q3_per_frame,
            'E2E': e2e_vals,
            'RMSF': rmsf_data
            }
        return avg_dict, full_dict

    except Exception as e:
        return f"Error in {filename}: {str(e)}"

def plot_metrics_vs_temp(df, base_name, temp_unit='K', ref_temp=False):
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
               ('Avg_RMSD', 'Root-mean-square-fluctuation (Å)')]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    
    for i, (col, label) in enumerate(metrics):
        axes[i].plot(summary['Temperature'], summary[col], marker='o', linestyle='-', color='teal')
    
        # temps = summary['Temperature'].values
        
        # # Explicitly set the positions AND the text labels
        # axes[i].set_xticks(temps)
        # axes[i].set_xticklabels([f"{t:.1f}" for t in temps])
        
        # axes[i].set_xlabel(f'Temperature ({temp_unit})')
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} vs T')
        
        if ref_temp:
            axes[i].axvline(ref_temp, color='red', ls='dashed', label='Reference')
        
        axes[i].grid(True, alpha=0.3)
        # Optional: rotate labels if they overlap
        axes[i].tick_params(axis='x', rotation=45) 

    plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig(f"{base_name}_temp_trends.png", dpi=300)
    print(f"Trend plots saved as {base_name}_temp_trends.png")

def plot_full_metrics(full_dict, output_name):
    """Take the metrics dataframe and plot output graphs"""
    # 1. Setup the figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=False)
    metrics = ['RMSD', 'Rg', 'Q3', 'E2E', 'RMSF']
    titles = ['RMSD Over Time', 'Radius of Gyration (Rg) Over Time', 
              'Q3 Folding Fraction Over Time', 'End-to-End Distance (E2E) Over Time', 
              'RMSF per Residue']
    
    # 2. Setup Color Mapping based on Temperature
    temps = [val['Temperature'] for val in full_dict.values()]
    norm = colors.Normalize(vmin=min(temps), vmax=max(temps))
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')

    # 3. Plot data for each state
    for state_id, data in full_dict.items():
        temp = data['Temperature']
        color = mapper.to_rgba(temp)
        
        for i, metric in enumerate(metrics):
            y_values = data[metric]
            # Use range(len()) as x-axis (Frame index or Residue index)
            x_values = range(len(y_values))
            
            axes[i].plot(x_values, y_values, color=color, alpha=0.6, label=f"{temp}K" if i == 0 else "")
            axes[i].set_ylabel(metric)
            axes[i].set_title(titles[i])

    # 4. Refine layout
    axes[-1].set_xlabel("Index (Frame / Residue)")
    
    # Add a colorbar to show the Temperature scale
    cbar = fig.colorbar(mapper, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Temperature (K)')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for colorbar
    plt.savefig(f"{output_name}_structural_summary.png", dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="REMD Data Analysis: Metrics vs Temperature")
    parser.add_argument("-i", "--input_dir",
                        required=True,
                        help="Path to directory containing state all-atom PDBs")
    parser.add_argument("-tm", "--temp_map",
                        required=True,
                        help="JSON file mapping state IDs to Temperatures")
    parser.add_argument("-n","--name", required=True,
                        help="File prefix to save files with")
    parser.add_argument("-r",
                        "--ref",
                        default="native_structure.pdb",
                        help="PDB for alignment and RMSD calculations")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=mp.cpu_count())
    parser.add_argument("-ct",
                        "--convert_temp",
                        type=bool, default=False,
                        help="Set to True to convert temperature units from K to C by subtracting 273")
    parser.add_argument("-rt",
                    "--ref_temp",
                    default=False,
                    help="Value to use for reference temperatures in plots")
    args = parser.parse_args()

    # Load Temperature Map
    with open(args.temp_map, 'r') as f:
        temp_map = json.load(f)

    temp_unit = 'K' # default unit is Kelvin
    if args.convert_temp:
        # Remove 273 to convert to celsius.
        for id, temp in temp_map.items():
            temp_map[id] = temp - 273
        temp_unit = 'C'

    files = sorted(glob.glob(os.path.join(args.input_dir, "state_*_AA.pdb")))
    selection = "protein and name CA"
    
    if not files:
        print("No files found matching the pattern state_*_AA.pdb")
        return

    print(f"Processing {len(files)} files on {args.jobs} cores...")
    
    with mp.Pool(processes=args.jobs) as pool:
        task_args = [(f, args.ref, selection, temp_map) for f in files]
        results_list = pool.starmap(process_state_file, task_args)

    valid_results = [res for res in results_list if isinstance(res, tuple)]
    errors = [res for res in results_list if isinstance(res, str)]

    for err in errors:
        print(f"Worker Error: {err}")

    if valid_results:
        # Extract just the avg_dicts for the DataFrame
        avg_list = [res[0] for res in valid_results]
        df = pd.DataFrame(avg_list)
        df.to_csv(f"{args.name}_temperature_summary.csv", index=False)
        
        plot_metrics_vs_temp(df, args.name, temp_unit=temp_unit, ref_temp=float(args.ref_temp))

        # 2. Extract just the full_dicts for the JSON and the full plots
        final_full_map = {res[1]['State']: res[1] for res in valid_results if 'State' in res[1]}
        plot_full_metrics(final_full_map, args.name)

        with open(f"{args.name}_full_data.json", "w") as out_dict:
            json.dump(final_full_map, out_dict, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    main()
