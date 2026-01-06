import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import argparse
import glob
import multiprocessing as mp
import tqdm
from MDAnalysis.analysis import rms, distances
from MDAnalysis.analysis.dssp import DSSP

# Prevent the monitor thread from starting
tqdm.tqdm.monitor_interval = 0

# --- Worker Function ---
def process_state_file(f, ref_file, selection, base_output_name):
    tag = os.path.basename(f).replace(".pdb", "")
    try:
        # 1. Load and Align
        ref = mda.Universe(ref_file)
        u = mda.Universe(f)
        if not u.select_atoms("all"): return None

        # Global Alignment to Reference for RMSD
        aligner = rms.RMSD(u, ref, select=selection).run()
        rmsd_vals = aligner.results.rmsd[:, 2]
        
        # 2. Structural Metrics
        protein = u.select_atoms('protein')
        # Radius of Gyration
        rg_vals = [protein.radius_of_gyration() for ts in u.trajectory]
        
        # RMSF (Per residue)
        u.trajectory.rewind()
        R = rms.RMSF(protein.select_atoms(selection)).run()
        
        # Q3 Secondary Structure (H, E, G content)
        first_res, last_res = protein.residues[0].resid, protein.residues[-1].resid
        q3_avg = 0
        try:
            # Select backbone atoms excluding termini for DSSP stability
            backbone = u.select_atoms(f"protein and (name N or name CA or name C or name O)")
            dssp = DSSP(backbone).run()
            q3_per_frame = [np.sum(np.isin(frame, ['H', 'E', 'G'])) / len(backbone.residues) for frame in dssp.results.dssp]
            q3_avg = np.mean(q3_per_frame)
        except: pass

        # End-to-End Distance
        e2e_avg = 0
        try:
            n_term = u.select_atoms(f'name N and resnum {first_res}')
            c_term = u.select_atoms(f'name CA and resnum {last_res}')
            e2e_vals = [distances.distance_array(n_term.positions, c_term.positions)[0][0] for ts in u.trajectory]
            e2e_avg = np.mean(e2e_vals)
        except: pass

        # 3. FEL Data (Segmented RMSD)
        n_res = len(u.residues)
        split = int(np.ceil(0.6 * n_res))
        sel_a = f"resid 1:{split} and name CA"
        sel_b = f"resid {n_res-split+1}:{n_res} and name CA"
        
        rmsd_a = rms.RMSD(u, u, select=sel_a).run().results.rmsd[:, 2]
        rmsd_b = rms.RMSD(u, u, select=sel_b).run().results.rmsd[:, 2]
        fel_data = np.column_stack((rmsd_a, rmsd_b))

        u.trajectory.close()
        ref.trajectory.close()

        return {
            'tag': tag,
            'summary': {
                'Trajectory': tag,
                'Avg_RMSD': np.mean(rmsd_vals),
                'Avg_Rg': np.mean(rg_vals),
                'Avg_Q3': q3_avg,
                'Avg_E2E': e2e_avg
            },
            'fel_data': fel_data
        }

    except Exception as e:
        return f"Error in {tag}: {str(e)}"

def generate_summary_heatmap(all_fel_arrays, base_name):
    """Combines all trajectory data into one global Free Energy Landscape."""
    combined_data = np.vstack(all_fel_arrays)
    
    T = 300 
    R_const = 8.31446 / 1000.0
    kT = R_const * T
    
    counts, x_edges, y_edges = np.histogram2d(combined_data[:, 0], combined_data[:, 1], bins=50)
    P = counts.T / np.sum(counts)
    P[P < 1e-10] = 1e-10
    G = -kT * np.log(P / np.max(P))
    
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0)
    cp = plt.contourf(X, Y, G, levels=20, cmap='viridis_r')
    plt.colorbar(cp, label='Delta G (kJ/mol)')
    plt.xlabel('RMSD Segment A (Å)')
    plt.ylabel('RMSD Segment B (Å)')
    plt.title('Combined Free Energy Landscape (All States)')
    plt.savefig(f"{base_name}_Combined_FEL.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel Analysis of State PDBs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref", default="native_structure.pdb")
    parser.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "state_*_AA.pdb")))
    selection = "protein and name CA"
    
    if not files:
        print("No files found matching pattern.")
        return

    print(f"Processing {len(files)} files...")
    
    results_list = []
    with mp.Pool(processes=args.jobs) as pool:
        task_args = [(f, args.ref, selection, args.name) for f in files]
        for res in tqdm(pool.starmap(process_state_file, task_args), total=len(task_args)):
            results_list.append(res)

    # 4. Collate Results
    summaries = []
    fel_arrays = []
    
    for item in results_list:
        if isinstance(item, dict):
            summaries.append(item['summary'])
            fel_arrays.append(item['fel_data'])
        else:
            print(item)

    # Save CSV Summary
    pd.DataFrame(summaries).to_csv(f"{args.name}_summary.csv", index=False)
    
    # Generate Combined Heatmap
    if fel_arrays:
        generate_summary_heatmap(fel_arrays, args.name)
        print(f"Analysis complete. Summary and Combined Heatmap saved.")

if __name__ == "__main__":
    main()
