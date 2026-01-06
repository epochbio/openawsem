import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob
import multiprocessing as mp
from tqdm import tqdm
from MDAnalysis.analysis import rms, distances
from MDAnalysis.analysis.dssp import DSSP

# --- Worker Function ---
def process_state_file(f, ref_file, selection, base_output_name):
    """
    Handles the full analysis pipeline for a single PDB file.
    Executed in parallel.
    """
    tag = os.path.basename(f).replace(".pdb", "")
    try:
        # 1. Load and Align
        ref = mda.Universe(ref_file)
        u = mda.Universe(f)
        
        if not u.select_atoms("all"):
            return None

        # Alignment to Reference
        aligner = rms.RMSD(u, ref, select=selection).run()
        total_rmsd = aligner.results.rmsd[:, 2].mean()

        # 2. Calculate Metrics
        protein_atoms = u.select_atoms('protein')
        
        # Radius of Gyration
        rg_values = [protein_atoms.radius_of_gyration() for ts in u.trajectory]
        
        # RMSF
        u.trajectory.rewind()
        R = rms.RMSF(protein_atoms, select=selection).run()
        
        # Q3 Secondary Structure
        first_res, last_res = protein_atoms.residues[0].resid, protein_atoms.residues[-1].resid
        try:
            # DSSP requires specific backbone atoms
            backbone_atoms = u.select_atoms(f"protein and (name N or name CA or name C or name O)")
            dssp_analysis = DSSP(backbone_atoms).run()
            q3_values = [np.sum(np.isin(frame, ['H', 'E', 'G'])) / len(backbone_atoms.residues) 
                         for frame in dssp_analysis.results.dssp]
            avg_q3 = np.mean(q3_values)
        except:
            avg_q3 = 0

        # End-to-End Distance
        try:
            n_term = u.select_atoms(f'name N and resnum {first_res}')
            c_term = u.select_atoms(f'name CA and resnum {last_res}')
            e2e = [distances.distance_array(n_term.positions, c_term.positions)[0][0] for ts in u.trajectory]
            avg_e2e = np.mean(e2e)
        except:
            avg_e2e = 0

        # 3. FEL Segmented RMSD
        n_res = len(u.residues)
        split_a = int(np.ceil(0.6 * n_res))
        split_b = n_res - int(np.floor(0.6 * n_res)) + 1
        
        sel_a = f"resid 1:{split_a} and name CA"
        sel_b = f"resid {split_b}:{n_res} and name CA"
        
        rmsd_a = rms.RMSD(u, u, select=sel_a).run().results.rmsd[:, 2]
        rmsd_b = rms.RMSD(u, u, select=sel_b).run().results.rmsd[:, 2]
        
        # Plotting FEL
        plot_fel(np.column_stack((rmsd_a, rmsd_b)), tag, base_output_name)

        # Return summary dictionary
        return {
            'Trajectory': tag,
            'Avg_RMSD': total_rmsd,
            'Avg_Rg': np.mean(rg_values),
            'Avg_Q3': avg_q3,
            'Avg_E2E': avg_e2e
        }

    except Exception as e:
        return f"Error in {tag}: {str(e)}"

def plot_fel(rmsd_data, name_tag, output_name):
    kT = 8.31446 / 1000.0 * 300
    counts, x_edges, y_edges = np.histogram2d(rmsd_data[:, 0], rmsd_data[:, 1], bins=50)
    P = counts.T / np.sum(counts)
    P[P < 1e-10] = 1e-10
    G = -kT * np.log(P / P.max())
    
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0)
    plt.contourf(X, Y, G, levels=20, cmap='viridis_r')
    plt.colorbar(label='Delta G (kJ/mol)')
    plt.xlabel('RMSD Segment A (Å)')
    plt.ylabel('RMSD Segment B (Å)')
    plt.title(f'FEL - {name_tag}')
    plt.savefig(f"{output_name}_FEL_{name_tag}.png")
    plt.close()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Analyze State PDBs in Parallel.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref", default="native_structure.pdb")
    parser.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    state_files = sorted(glob.glob(os.path.join(args.input_dir, "state_*_AA.pdb")))
    selection = "protein and name CA"
    
    if not state_files:
        print("No files found.")
        return

    # Prepare Arguments
    task_args = [(f, args.ref, selection, args.name) for f in state_files]

    # Parallel Pool
    print(f"Processing {len(state_files)} files using {args.jobs} cores...")
    
    all_summary = []
    with mp.Pool(processes=args.jobs) as pool:
        # tqdm wraps the imap_unordered for a live progress bar
        for result in tqdm(pool.starmap(process_state_file, task_args), total=len(task_args)):
            if isinstance(result, dict):
                all_summary.append(result)
            else:
                print(f"\n{result}") # Print errors if they occurred

    # Save Results
    if all_summary:
        df = pd.DataFrame(all_summary)
        df.to_csv(f"{args.name}_summary.csv", index=False)
        print(f"\nAnalysis complete. Results saved to {args.name}_summary.csv")

if __name__ == "__main__":
    main()
