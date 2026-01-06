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
    description="Convert openASWEM PDB to all-atom backbone representation."
)
parser.add_argument("pdb", help="The path to the reference openawsem pdb")
parser.add_argument("-f", "--fasta", default="./crystal_structure.fasta", help="Reference fasta for PDB mapping. Default is ./crystal_structure.fasta")

args = parser.parse_args()
pdb = args.pdb

seq_dic = get_seq_dic(fasta=args.fasta)

file_name = pdb.split('.')[0]
traj = md.load(top=pdb)
print(f"Successfully loaded {file_name}.pdb'")

traj.save_pdb(f"{file_name}_reference_AA.pdb")

convert_openMM_to_standard_pdb(
    fileName=f"{file_name}_reference_AA.pdb",
    seq_dic=seq_dic,
)
print(f"Successfully converted reference all-atom format and wrote {file_name}_reference_AA.pdb")


