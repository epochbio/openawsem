import os
import glob
import mdtraj as md
import argparse
import sys

try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.insert(0, OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION." \
    "\n Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    exit()

from helperFunctions.myFunctions import \
    convert_openMM_to_standard_pdb, get_seq_dic

parser = argparse.ArgumentParser(
    description="Convert openMM output to a standard multi-frame PDB."
)
parser.add_argument("folder", help="The path to the folder with the replica dcd files")
parser.add_argument("-f", "--fasta", default="./crystal_structure.fasta", help="Reference fasta for PDB mapping. Default is ./crystal_structure.fasta")
parser.add_argument("-r", "--ref", default="crystal_structure-openmmawsem.pdb")
parser.add_argument("-dp",
                    "--dcd_pattern",
                    default="*.dcd",
                    help="pattern to search for DCD files for. Default='*.dcd'. Additional '.'s in the name may cause errors")

args = parser.parse_args()

folder = args.folder
ref = args.ref

seq_dic = get_seq_dic(fasta=args.fasta)
pattern = args.dcd_pattern

dcd_files = glob.glob(f'{folder}/{pattern}')

# Replace filenames with your actual files
for file in dcd_files:
    file_name = file.split('.')[0]
    traj = md.load(f'{file_name}.dcd', top=ref)
    print(f"Successfully loaded {file_name}.dcd'")
    traj.save_pdb(f"{file_name}_OA.pdb")
    print(f"Wrote coarse-grained PDB {file_name}_OA.pdb")
    traj.save_pdb(f"{file_name}_AA.pdb")

    convert_openMM_to_standard_pdb(
        fileName=f"{file_name}_AA.pdb",
        seq_dic=seq_dic,
    )
    print(f"Successfully converted to all-atom format and wrote {file_name}_AA.pdb")


