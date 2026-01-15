import os
import io
import sys
import glob
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import multiprocessing as mp
from Bio.PDB.PDBParser import PDBParser
from typing import Tuple, List, Dict

# Import necessary helper functions
try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.insert(0, OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.")
    exit()

from helperFunctions.myFunctions import compute_native_contacts


def getAllFrames(movieLocation: str) -> Tuple[List[str], int, int]:
    with open(movieLocation) as f:
        a = f.readlines()
    n = len(a)
    model_title_index_list = [i for i, line in enumerate(a) if line.startswith("MODEL")]
    if not model_title_index_list:
        print(f"Error: No 'MODEL' records found in {movieLocation}")
        return [], 0, 0
    model_title_index_list.append(n)
    check_array = np.diff(model_title_index_list)
    size = check_array[0]
    return a, n, size

def get_contact_matrix(coords: np.ndarray, mask: np.ndarray, DISTANCE_CUTOFF: float = 9.5) -> np.ndarray:
    a = coords[:, np.newaxis]
    dis = np.sqrt(np.sum((a - coords)**2, axis=2))
    contacts_in_frame = (dis < DISTANCE_CUTOFF).astype(int)
    return contacts_in_frame * mask

def get_reference_native_contacts(native_pdb_file: str, MAX_OFFSET: int = 4, DISTANCE_CUTOFF: float = 9.5) -> np.ndarray:
    p = PDBParser()
    s = p.get_structure("native_state", native_pdb_file)
    coords = extract_coords_from_structure(s)
    return compute_native_contacts(coords, MAX_OFFSET, DISTANCE_CUTOFF)

def extract_coords_from_structure(s: object) -> np.ndarray:
    chains = s[0].get_list()
    coords = []
    for chain in chains:
        for res in chain:
            is_regular_res = res.has_id('CA') and res.has_id('O')
            if (res.get_resname() == 'GLY') and res.has_id('CA'):
                coords.append(res['CA'].get_coord())
            elif is_regular_res and res.has_id('CB'):
                coords.append(res['CB'].get_coord())
            elif is_regular_res and res.has_id('CA'):
                coords.append(res['CA'].get_coord())
    return np.array(coords)


def process_trajectory_for_contact_frequency(movie_file_path: str, N_native: np.ndarray, temp_map: dict):
    """Processes trajectory and returns summary statistics instead of plotting."""
    try:
        base_name = os.path.basename(movie_file_path).replace(".pdb", "")
        state_id = base_name.split('_')[1]
        temperature = temp_map.get(state_id) or temp_map.get(str(state_id))
        if temperature is None: return None
    except: return None

    N_residues = N_native.shape[0]
    native_counts = np.zeros((N_residues, N_residues), dtype=np.int32)
    all_counts = np.zeros((N_residues, N_residues), dtype=np.int32)
    total_frames = 0
    p = PDBParser()

    allFrames, n, size = getAllFrames(movie_file_path)
    if not allFrames: return None
    num_of_frames = int(n / size)

    for frame_index in range(num_of_frames):
        oneFrame = allFrames[size*frame_index : size*(frame_index+1)]
        try:
            f = io.StringIO("".join(oneFrame))
            s = p.get_structure("frame", f)
            coords = extract_coords_from_structure(s)
            
            native_counts += get_contact_matrix(coords, N_native)
            all_counts += get_contact_matrix(coords, np.ones((N_residues, N_residues), dtype=int))
            total_frames += 1
        except: continue

    if total_frames == 0: return None

    # Calculate average frequency matrices
    freq_native = native_counts / total_frames
    freq_all = all_counts / total_frames
    
    # Calculate average number of contacts per frame
    avg_num_native = np.sum(native_counts) / (2 * total_frames) # Divide by 2 for symmetric matrix
    avg_num_all = np.sum(all_counts) / (2 * total_frames)

    return {
        "temp": temperature,
        "freq_native": freq_native,
        "freq_all": freq_all,
        "avg_num_native": avg_num_native,
        "avg_num_all": avg_num_all
    }


def plot_contact_totals(results: list,
                        name:str,
                        output_dir: str, ref_temp: float):
    """Plots Number of Contacts vs Temperature."""
    results.sort(key=lambda x: x['temp'])
    temps = [r['temp'] for r in results]
    native_totals = [r['avg_num_native'] for r in results]
    all_totals = [r['avg_num_all'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(temps, native_totals, 'o-', color='blue')
    ax1.set_title("Native Contacts")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Count")

    ax2.plot(temps, all_totals, 'o-', color='red')
    ax2.set_title("Total Contacts")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Count")

    for ax in [ax1, ax2]:
        if ref_temp is not None:
            ax.axvline(x=float(ref_temp), color='green', linestyle='--', label=f'Ref Temp: {ref_temp}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_contacts_vs_temperature.png"))
    plt.close()

def plot_single_summed_heatmap(results: list,
                               name: str,
                               output_dir: str, contact_type: str):
    """Plots a single heatmap summing frequency across all temperatures."""
    key = 'freq_native' if contact_type == 'Native' else 'freq_all'
    
    # Initialize zero matrix based on first result
    summed_matrix = np.zeros_like(results[0][key])
    
    for res in results:
        summed_matrix += res[key]
    
    # Normalize by the number of temperatures to get the global average frequency
    avg_matrix = summed_matrix / len(results)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_matrix, cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(label='Average Contact Frequency (Across All Temps)')
    plt.title(f'Global Average {contact_type} Contact Frequency')
    plt.xlabel('Residue i')
    plt.ylabel('Residue j')
    
    plt.savefig(os.path.join(output_dir, f"{name}_global_average_{contact_type.lower()}_heatmap.png"), dpi=300)
    plt.close()

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path")
    parser.add_argument("native_pdb")
    parser.add_argument("-n","--name", required=True,
                        help="File prefix to save files with")
    parser.add_argument("-o", "--output_path", default="native_contacts")
    parser.add_argument("-pp", "--pdb_pattern", default="state*AA.pdb")
    parser.add_argument("-tm", "--temp_map", required=True)
    parser.add_argument("-ct", "--convert_temp", action='store_true')
    parser.add_argument("-rt", "--ref_temp", default=None)
    parser.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())

    args = parser.parse_args()
    
    with open(args.temp_map, 'r') as f:
        temp_map = json.load(f)
    if args.convert_temp:
        temp_map = {k: v - 273 for k, v in temp_map.items()}

    N_native_reference = get_reference_native_contacts(args.native_pdb)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pdb_files = glob.glob(os.path.join(args.folder_path, args.pdb_pattern))
    
    print(f"Processing {len(pdb_files)} files...")
    task_args = [(f, N_native_reference, temp_map) for f in pdb_files]
    
    with mp.Pool(processes=args.jobs) as pool:
        results = pool.starmap(process_trajectory_for_contact_frequency, task_args)
    
    # Filter out None results (where state ID wasn't in map)
    results = [r for r in results if r is not None]

    if not results:
        print("No valid results found.")
        return

    # Generate the new combined plots
    # print("Generating summary plots...")
    # plot_combined_heatmaps(results, args.output_path, "Native")
    # plot_combined_heatmaps(results, args.output_path, "All")

    print("Generating Global Average Heatmaps...")
    plot_single_summed_heatmap(results, args.name, args.output_path, "Native")
    plot_single_summed_heatmap(results, args.name, args.output_path, "All")

    plot_contact_totals(results, args.name, args.output_path, args.ref_temp)
    print(f"Done. Outputs saved to {args.output_path}")

if __name__ == "__main__":
    main()
