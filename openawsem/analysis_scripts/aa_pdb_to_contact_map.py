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
from typing import Tuple, List

# Import necessary helper functions from the helper functions folder
try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.insert(0, OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.\n Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    exit()

from helperFunctions.myFunctions import compute_native_contacts


def getAllFrames(movieLocation: str
                 ) -> Tuple[List[str], int, int]:
    # ... (function body as in original script) ...
    # Reads file and determines 'size' (lines per frame)
    location = movieLocation
    with open(location) as f:
        a = f.readlines()
    n = len(a)
    model_title_index_list = []
    for i in range(n):
        if len(a[i]) >= 5 and a[i][:5] == "MODEL":
            model_title_index = i
            model_title_index_list.append(model_title_index)
    
    if not model_title_index_list:
        print("Error: No 'MODEL' records found. Is this a multi-frame PDB?")
        exit()
        
    model_title_index_list.append(n)
    
    # Check for consistent block size
    check_array = np.diff(model_title_index_list)
    if np.allclose(check_array, check_array[0]) or \
       (np.allclose(check_array[:-1], check_array[0]) and check_array[-1] == check_array[0] + 1):
        size = check_array[0]
    else:
        print("!!!! Someting is wrong with frame size consistency !!!!")
        print(check_array)
        size = check_array[0] # Try to continue with the first block size
    return a, n, size

def get_contact_matrix(coords: np.ndarray, 
                       native_contacts: np.ndarray, # This is N_native (binary)
                       DISTANCE_CUTOFF: float = 9.5
                       ) -> np.ndarray:
    """
    MODIFIED: Compute and return the matrix of native contacts for the current frame.
    """
    native_coords = np.array(coords)
    a= native_coords[:,np.newaxis]
    dis = np.sqrt(np.sum((a - native_coords)**2, axis=2))
    
    contacts_in_frame = dis < DISTANCE_CUTOFF
    
    # Filter to only keep *native* contacts
    current_native_contacts = contacts_in_frame * native_contacts  
    
    # Return the binary matrix (0 or 1)
    return current_native_contacts.astype("int")

def get_reference_native_contacts(native_pdb_file: str, MAX_OFFSET: int = 4, DISTANCE_CUTOFF: float = 9.5) -> np.ndarray:
    """Loads the reference PDB and computes the single, static N_native matrix."""
    p = PDBParser()
    s = p.get_structure("native_state", native_pdb_file)
    chains = s[0].get_list()

    native_coords = []
    # Collect coordinates (using CA/CB logic) similar to your localQ_init
    for chain in chains:
        for res in chain:
            is_regular_res = res.has_id('CA') and res.has_id('O')
            res_id = res.get_id()[0]
            if (res.get_resname()=='GLY') and res.has_id('CA'):
                native_coords.append(res['CA'].get_coord())
            elif is_regular_res and res.has_id('CB'):
                native_coords.append(res['CB'].get_coord())
            elif is_regular_res and res.has_id('CA'):
                native_coords.append(res['CA'].get_coord())

    if not native_coords:
        raise ValueError(f"Could not extract coordinates from reference PDB: {native_pdb_file}")
        
    # Use your provided function
    native_contacts_table = compute_native_contacts(native_coords, MAX_OFFSET, DISTANCE_CUTOFF)
    
    return native_contacts_table # This is the N_native binary matrix

def plot_heatmap(matrix: np.ndarray, base_name: str, output_dir: str):
    """Plots the lower triangle of the contact frequency map."""
    
    N = matrix.shape[0]
    
    plt.figure(figsize=(8, 8))
    
    # Define a colormap for N_ij / N_max
    cmap = 'coolwarm' # Use a simple continuous map for now
    norm = mcolors.Normalize(vmin=0, vmax=1) # Normalized from 0 (never present) to 1 (always present)
    
    # Plot the masked data
    plt.imshow(matrix, cmap=cmap, norm=norm, origin='upper')
    
    # Add a color bar
    cbar = plt.colorbar(shrink=0.8, pad=0.05)
    cbar.set_label(r'$N_{ij} / N_{max}$ (Contact Frequency)') 
    
    plt.xlabel('Residue i')
    plt.ylabel('Residue j')
    plt.title(f'Contact Frequency Map ({base_name} ˚C)')
    
    plt.savefig(os.path.join(output_dir, f"{base_name}_contact_frequency.png"), dpi=300)
    plt.close()


def extract_coords_from_structure(s: object) -> np.ndarray:
    """Helper function to extract coordinates from a single PDB frame structure."""
    chains = s[0].get_list()
    coords = []
    for chain in chains:
        for res in chain:
            is_regular_res = res.has_id('CA') and res.has_id('O')
            res_id = res.get_id()[0]
            
            if (res.get_resname()=='GLY') and res.has_id('CA'):
                coords.append(res['CA'].get_coord())
            elif is_regular_res and res.has_id('CB'):
                coords.append(res['CB'].get_coord())
            elif is_regular_res and res.has_id('CA'):
                 coords.append(res['CA'].get_coord())
    return np.array(coords)


def process_trajectory_for_contact_frequency(movie_file_path: str, output_dir: str, N_native: np.ndarray, temp_map: dict):
    """
    Processes a multi-frame PDB, calculates the persistence frequency of each native 
    contact over the entire trajectory, and plots the result.
    """
    
    try:
        base_name = os.path.basename(movie_file_path).replace(".pdb", "")
        state_id = base_name.split('_')[1]
        temperature = temp_map.get(state_id) or temp_map.get(int(state_id))

        if temperature is None:
            return f"Error: State {state_id} not found in JSON temp map."
    except:
        return f"Error: Could not parse state ID from {movie_file_path}"
    
    # 1. Initialization
    N_residues = N_native.shape[0]
    native_count_matrix = np.zeros((N_residues, N_residues), dtype=np.int32)
    all_count_matrix = np.zeros((N_residues, N_residues), dtype=np.int32)
    total_frames = 0
    p = PDBParser()

    print(f"Starting simple contact frequency analysis for: {base_name}," \
          f" Temperature: {temperature}...")

    # 2. Iterate through frames
    try:
        allFrames, n, size = getAllFrames(movie_file_path) # Uses your existing frame splitting logic
        num_of_frames = int(n / size)
    except Exception as e:
        print(f"Error reading frames from {movie_file_path}: {e}")
        return

    for frame_index in range(num_of_frames):
        
        oneFrame = allFrames[size*frame_index : size*(frame_index+1)]
        
        if not oneFrame:
            continue
            
        try:
            # Parse the frame text into a Biopython Structure object
            f = io.StringIO("".join(oneFrame))
            s = p.get_structure("frame", f) 
            
            # Extract coordinates
            frame_coords = extract_coords_from_structure(s)
            
            # Calculate the contact matrix for the current frame
            # N_native is passed as the mask/filter
            current_contact_matrix = get_contact_matrix(frame_coords, N_native)
            current_all_matrix = get_contact_matrix(frame_coords, np.ones((N_residues, N_residues), dtype=np.int32))
            
            # Accumulate the count
            native_count_matrix += current_contact_matrix
            all_count_matrix += current_all_matrix
            total_frames += 1
        
        except Exception as e:
            print(f"Warning: Skipping Frame {frame_index} in {base_name} due to error: {e}")

    print(f"Accumulation complete for {base_name}. Processed {total_frames} frames.")

    # 3. Normalize and Plot
    if total_frames > 0:
        # Normalized matrix = N_ij / Total Frames
        normalized_matrix = native_count_matrix / total_frames
        normalized_all_matrix = all_count_matrix / total_frames
    else:
        normalized_matrix = native_count_matrix
        normalized_all_matrix = all_count_matrix

    plot_heatmap(normalized_matrix, temperature, output_dir)
    plot_heatmap(normalized_all_matrix, f'{temperature}_all', output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate, plot, and save native contact frequency for all multi-frame PDBs matching replica*AA.pdb or an alternative pdb pattern, in a specified folder."
    )
    parser.add_argument("folder_path",
                        help="The folder containing the state*AA.pdb multi-frame files.")
    parser.add_argument("native_pdb",
                        help="The path to the single reference native PDB file (e.g., 2xov.pdb).")
    parser.add_argument("-j", "--jobs",
                        type=int,   
                        default=mp.cpu_count(),
                        help="Number of parallel processes to use (default: max available CPUs).")
    parser.add_argument("-o", "--output_path",
                         default="native_contacts",
                           help="Output folder to save plots, default native_contacts")
    parser.add_argument("-pp",
                        "--pdb_pattern",
                        default="state*AA.pdb",
                        help="pattern to search for PDBs for. Default='state*AA.pdb'")
    parser.add_argument("-tm", "--temp_map",
                        required=True,
                        help="JSON file mapping state IDs to Temperatures")
    parser.add_argument("-ct",
                        "--convert_temp",
                        type=bool, default=False,
                        help="Set to True to convert temperature units from K to C by subtracting 273")
    parser.add_argument("-rt",
                    "--ref_temp",
                    default=False,
                    help="Value to use for reference temperatures in plots")

    args = parser.parse_args()
    input_folder = args.folder_path
    native_pdb_file = args.native_pdb
    num_processes = args.jobs
    output_folder = args.output_path
    pdb_pattern = args.pdb_pattern

        # Load Temperature Map
    with open(args.temp_map, 'r') as f:
        temp_map = json.load(f)

    temp_unit = 'K' # default unit is Kelvin
    if args.convert_temp:
        # Remove 273 to convert to celsius.
        for id, temp in temp_map.items():
            temp_map[id] = temp - 273
        temp_unit = 'C'

    # --- 1. INITIALIZATION: Get the static N_native matrix ---
    try:
        # Calculates the native reference
        N_native_reference = get_reference_native_contacts(native_pdb_file)
        print(f"Reference N_native matrix (Size: {N_native_reference.shape[0]}x{N_native_reference.shape[0]}) created successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not create reference native contacts from {native_pdb_file}. {e}")
        sys.exit(1)


    # 2. Create the output folder and Discover PDB files 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Look for all atom pdbs.
    pattern = os.path.join(input_folder, pdb_pattern)
    pdb_files = glob.glob(pattern)
    
    if not pdb_files:
        print(f"No files found matching '{pattern}'.")
        sys.exit(0)
        
    print(f"Found {len(pdb_files)} multi-frame PDB files to process.")

    # 3. Use multiprocessing
    print(f"Starting analysis with {num_processes} parallel processes...")
    try:
        # Prepare arguments: (file_path, output_dir, N_native_reference)
        # N_native_reference is the same static array for every process
        task_args = [(f, output_folder, N_native_reference, temp_map) for f in pdb_files]
        
        with mp.Pool(processes=num_processes) as pool:
            # Note: numpy arrays are efficiently pickled/passed to the pool workers
            pool.starmap(process_trajectory_for_contact_frequency, task_args)
            
        print("\nAll contact frequency maps processed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during multiprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

