import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
import glob
from MDAnalysis.analysis import rms, distances
from MDAnalysis.analysis.dssp import DSSP

def load_and_align_trajectory(trajectory_file, reference_file, selection):
    if not os.path.exists(trajectory_file):
        print(f"Warning: File not found: {trajectory_file}. Skipping.")
        return None
    ref = mda.Universe(reference_file)
    mobile = mda.Universe(trajectory_file)
    if not mobile.select_atoms("all"):
        return None
    aligner = rms.RMSD(mobile, ref, select=selection, groupselections=[selection, selection])
    aligner.run()
    return mobile, aligner.results.rmsd

def calculate_metrics(u_aligned, selection, file_tag):
    data = {}
    u_aligned.trajectory.rewind()
    protein_atoms = u_aligned.select_atoms('protein')
    
    # 1. Radius of Gyration
    data['Rg'] = [protein_atoms.radius_of_gyration() for ts in u_aligned.trajectory]
    u_aligned.trajectory.rewind()

    # 2. RMSF
    R = rms.RMSF(protein_atoms, select=selection).run()
    data['RMSF_Per_Atom'] = pd.DataFrame({'Residue_ID': protein_atoms.resids, 'RMSF': R.results.rmsf})
    u_aligned.trajectory.rewind()

    # 3. Q3 Secondary Structure
    try:
        first_res, last_res = protein_atoms.residues[0].resid, protein_atoms.residues[-1].resid
        residue_range = f"protein and not (resnum {first_res} or resnum {last_res}) and (name N or name CA or name C or name O)"
        backbone_atoms = u_aligned.select_atoms(residue_range)
        dssp_analysis = DSSP(backbone_atoms).run()
        q3_values = [np.sum(np.isin(frame, ['H', 'E', 'G'])) / len(backbone_atoms.residues) for frame in dssp_analysis.results.dssp]
        data['Q3_Content'] = q3_values
    except Exception as e:
        data['Q3_Content'] = []

    # 4. End-to-End Distance
    try:
        n_term = u_aligned.select_atoms(f'name N and resnum {first_res + 1}')
        c_term = u_aligned.select_atoms(f'name CA and resnum {last_res - 1}')
        data['End_to_End_Distance'] = [distances.distance_array(n_term.positions, c_term.positions)[0][0] for ts in u_aligned.trajectory]
    except:
        data['End_to_End_Distance'] = []
    
    return data

def plot_fel(rmsd_data, name_tag, output_name):
    # Calculate FEL based on 2 overlapping segments (first 60% and last 60% of residues)
    # This logic mimics the "RMSD FEL.ipynb" segmenting approach
    T = 300 # Default temperature if not mapped
    R = 8.31446 / 1000.0
    kT = R * T
    
    counts, x_edges, y_edges = np.histogram2d(rmsd_data[:, 0], rmsd_data[:, 1], bins=50)
    P = counts.T / np.sum(counts)
    P[P < 1e-10] = 1e-10
    G = -kT * np.log(P / P.max())
    
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0)
    plt.contourf(X, Y, G, levels=np.arange(0, np.ceil(G.max()/5)*5, 5), cmap='viridis_r')
    plt.colorbar(label='Delta G (kJ/mol)')
    plt.xlabel('RMSD Segment A (Å)')
    plt.ylabel('RMSD Segment B (Å)')
    plt.title(f'Free Energy Landscape - {name_tag}')
    plt.savefig(f"{output_name}_FEL_{name_tag}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze State PDBs for structural metrics and FEL.")
    parser.add_argument("--input_dir", required=True, help="Path to folder containing state_*.pdb files")
    parser.add_argument("--name", required=True, help="Base name for output files")
    parser.add_argument("--ref", default="native_structure.pdb", help="Reference PDB file")
    args = parser.parse_args()

    # Filter for 'state' PDBs only, ignoring 'replica'
    state_files = sorted(glob.glob(os.path.join(args.input_dir, "state_*_AA.pdb")))
    selection = "protein and name CA"
    
    all_summary = []
    all_metrics_json = {}

    for f in state_files:
        tag = os.path.basename(f).replace(".pdb", "")
        print(f"Analyzing {tag}...")
        
        u, rmsd_array = load_and_align_trajectory(f, args.ref, selection)
        if u is None: continue
        
        # Calculate standard metrics
        metrics = calculate_metrics(u, selection, tag)
        
        # Save summary data
        all_summary.append({
            'Trajectory': tag,
            'Avg_RMSD': rmsd_array[:, 2].mean(),
            'Avg_Rg': np.mean(metrics['Rg']),
            'Avg_Q3': np.mean(metrics['Q3_Content']) if metrics['Q3_Content'] else 0,
            'Avg_E2E': np.mean(metrics['End_to_End_Distance']) if metrics['End_to_End_Distance'] else 0
        })
        
        # Segmented RMSD for FEL (approx 60% overlap logic from Notebook 2)
        n_res = len(u.residues)
        split_a, split_b = int(np.ceil(0.6 * n_res)), n_res - int(np.floor(0.6 * n_res)) + 1
        sel_a, sel_b = f"resid 1:{split_a} and name CA", f"resid {split_b}:{n_res} and name CA"
        
        rmsd_a = rms.RMSD(u, u.universe, select=sel_a).run().results.rmsd[:, 2]
        rmsd_b = rms.RMSD(u, u.universe, select=sel_b).run().results.rmsd[:, 2]
        plot_fel(np.column_stack((rmsd_a, rmsd_b)), tag, args.name)

    # Export Results
    pd.DataFrame(all_summary).to_csv(f"{args.name}_summary.csv", index=False)
    print(f"Analysis complete. Results saved with prefix: {args.name}")

if __name__ == "__main__":
    main()
