import os
import argparse
import sys
import io
from Bio.PDB.PDBParser import PDBParser
import matplotlib.pyplot as plt

try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.insert(0, OPENAWSEM_LOCATION)
    # print(OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.\n Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    exit()

from helperFunctions.myFunctions import *
from typing import List, Tuple
from Bio.PDB import Structure


parser = argparse.ArgumentParser(
    description="Show the contact map of defined frame."
)
parser.add_argument("openmm", help="The name of the OpenAWSEM output")
parser.add_argument("-f", "--frame", default=-1, help="Default is Last frame.(-1), frame=0 means the first frame")
args = parser.parse_args()

movieFile = args.openmm


def getAllFrames(movieLocation: str
                 ) -> Tuple[List[str], int, int]:
    """Extract all frames from a given movie file location.

    Args:
        movieLocation (str): The file path to the movie file.

    Returns:
        Tuple[List[str], int, int]: A tuple containing a list of all lines in the file,
                                    the total number of lines, and the size of each model block.
    """
    # movieLocation = "/Users/weilu/Research/examples/openMM_simulation/test_2/movie.pdb"
    location = movieLocation
    with open(location) as f:
        a = f.readlines()
    n = len(a)
    # get the position of every model title
    model_title_index_list = []
    for i in range(n):
        if len(a[i]) >= 5 and a[i][:5] == "MODEL":
            model_title_index = i
            model_title_index_list.append(model_title_index)
    model_title_index_list.append(n)
    check_array = np.diff(model_title_index_list)
    if np.allclose(check_array, check_array[0]):
        size = check_array[0]
    elif np.allclose(check_array[:-1], check_array[0]) and check_array[-1] == check_array[0] + 1:
        # this is ok. with extra "END"
        size = check_array[0]
    else:
        print("!!!! Someting is wrong  !!!!")
        print(check_array)

    return a, n, size

def get_contacts_table(s: Structure, 
                       MAX_OFFSET: int = 4, 
                       DISTANCE_CUTOFF: float = 9.5
                       ) -> np.ndarray:
    """Calculate the contact map table for a given structure.

    Args:
        s (Structure): The structure object from Biopython PDB parser.
        MAX_OFFSET (int): The maximum sequence separation for two residues to be considered in contact.
        DISTANCE_CUTOFF (float): The distance cutoff in Angstroms for two residues to be considered in contact.

    Returns:
        np.ndarray: A 2D numpy array representing the contact map.
    """
    chains = s[0].get_list()
    # import pdb file
    coords = []
    for chain in chains:
        dis = []
        all_res = []
        for res in chain:
            is_regular_res = res.has_id('CA') and res.has_id('O')
            res_id = res.get_id()[0]
            if (res.get_resname()=='GLY'):
                coords.append(res['CA'].get_coord())
            elif (res_id==' ' or res_id=='H_MSE' or res_id=='H_M3L' or res_id=='H_CAS') and is_regular_res:
                coords.append(res['CB'].get_coord())
            else:
                print('ERROR: irregular residue at %s!' % res)
                exit()
    contacts_table = compute_native_contacts(coords, MAX_OFFSET, DISTANCE_CUTOFF)
    return contacts_table


allFrames, n, size = getAllFrames(movieFile)
num_of_frames = int(n/size)
frame = args.frame
if frame < 0:
    frame = num_of_frames + frame
oneFrame = allFrames[size*frame:size*(frame+1)]



p = PDBParser()
f = io.StringIO("".join(oneFrame))
s = p.get_structure("test", f)

contacts = get_contacts_table(s)
plt.imshow(contacts, origin=[0,0])
plt.show()