import os
import glob
import mdtraj as md
import multiprocessing as mp
import argparse
import sys

# --- Environment Setup ---
try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.insert(0, OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.")
    print("Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    sys.exit(1)

from helperFunctions.myFunctions import \
    convert_openMM_to_standard_pdb, get_seq_dic

# --- Worker Function ---
def process_trajectory(dcd_file, ref_pdb, sequence_dict):
    """
    This function runs on a single CPU core. 
    It handles the loading, saving, and conversion for one file.
    """
    try:
        # Get the base name (e.g., 'movie' from 'movie.dcd')
        file_prefix = os.path.splitext(dcd_file)[0]
        
        # 1. Load the trajectory
        traj = md.load(dcd_file, top=ref_pdb)
        
        # 2. Save intermediate coarse-grained PDB
        oa_pdb = f"{file_prefix}_OA.pdb"
        traj.save_pdb(oa_pdb)
        
        # 3. Perform the OpenAWSEM conversion to standard PDB
        # Assuming this function takes (prefix, sequence_dict)
        convert_openMM_to_standard_pdb(file_prefix, sequence_dict)
        
        return f"Successfully processed: {dcd_file}"
    except Exception as e:
        return f"Error processing {dcd_file}: {str(e)}"

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Convert openMM output to a standard multi-frame PDB in parallel."
    )
    parser.add_argument("folder", help="The path to the folder with the replica dcd files")
    parser.add_argument("-f", "--fasta", default="./crystal_structure.fasta", help="Reference fasta.")
    parser.add_argument("-r", "--ref", default="crystal_structure-openmmawsem.pdb")
    parser.add_argument("-dp", "--dcd_pattern", default="*.dcd", help="Pattern for DCD files.")
    parser.add_argument("-j", "--jobs", type=int, default=mp.cpu_count(),
                        help="Number of parallel processes to use.")

    args = parser.parse_args()

    # Prepare file list
    pattern = os.path.join(args.folder, args.dcd_pattern)
    dcd_files = glob.glob(pattern)

    if not dcd_files:
        print(f"No files found matching pattern: {pattern}")
        return

    seq_dict = get_seq_dic(fasta=args.fasta)

    # Prepare arguments for the pool
    # Each tuple contains: (specific_dcd, reference_pdb, dictionary)
    task_args = [(f, args.ref, seq_dict) for f in dcd_files]

    print(f"Found {len(dcd_files)} files. Starting {args.jobs} parallel workers...")

    # 

    # Execute parallel tasks
    try:
        with mp.Pool(processes=args.jobs) as pool:
            # starmap allows passing multiple arguments to the function
            results = pool.starmap(process_trajectory, task_args)
        
        # Print summary of results
        for res in results:
            print(res)
            
        print("\nAll files processed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during multiprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
