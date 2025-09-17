# ----------------------------------------------------------------------
# Copyright (2010) Aram Davtyan and Garegin Papoian
#
# Papoian's Group, University of Maryland at Collage Park
# http://papoian.chem.umd.edu/
#
# Last Update: 07/08/2011
# -------------------------------------------------------------------------

from Bio.PDB.PDBParser import PDBParser

class Atom:

    def __init__(self, 
				 atom_no: int, 
				 atom_name: str, 
				 res_no: int, 
				 res_name: str,
				 xyz: tuple, 
				 desc: str = ''):
        """
        Initialize an Atom instance with a tuple containing x, y, z coordinates.

        Args:
            atom_no (int): Atom number.
            atom_name (str): Atom name.
            res_no (int): Residue number.
            res_name (str): Residue name.
            xyz (tuple): Tuple containing x, y, z coordinates.
            desc (str): Description or additional information. Defaults to an empty string.
        """
        self.atom_no = atom_no
        self.atom_name = atom_name
        self.res_no = res_no
        self.res_name = res_name
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]
        self.desc = desc

    def print_(self):
        """
        Print the atom details.
        """
        print(self.atom_no, self.atom_name, self.res_no, self.res_name, self.x, self.y, self.z, self.desc)

    def write_(self, f):
        """
        Write the atom details to a file in GRO format.

        Args:
            f: File object to write the atom details.
        """
        f.write( ("     "+str(self.res_no))[-5:] )
        f.write( ("     "+self.res_name)[-5:] )
        f.write( " " + (self.atom_name+"    ")[:4] )
        f.write( ("     "+str(self.atom_no))[-5:] )
        f.write( ("        "+str(round(self.x/10,3)))[-8:] )
        f.write( ("        "+str(round(self.y/10,3)))[-8:] )
        f.write( ("        "+str(round(self.z/10,3)))[-8:] )
        f.write("\n")


def Pdb2Gro(pdb_file: str, 
			gro_file: str, 
			ch_name: str
			) -> None:
	"""
	Convert a PDB file to a GRO file for a specified chain.

	Args:
		pdb_file: The path to the input PDB file.
		gro_file: The path where the output GRO file will be saved.
		ch_name: The name of the chain to be converted.
	"""
	p = PDBParser(PERMISSIVE=1, QUIET=True)

	pdb_id = pdb_file
	if pdb_file[-4:].lower()!=".pdb":
		pdb_file = pdb_file + ".pdb"
	if pdb_id[-4:].lower()==".pdb":
		pdb_id = pdb_id[:-4]

	output = gro_file

	s = p.get_structure(pdb_id, pdb_file)
	chains = s[0].get_list()

	if ch_name=='':
		ch_name = 'A'

	for chain in chains:
		if chain.get_id()==ch_name:
			ires = 0
			iatom = 0
			res_name = ""
			atoms = []
			for res in chain:
				is_regular_res = res.has_id('N') and res.has_id('CA') and res.has_id('C')
				res_id = res.get_id()[0]
				if (res_id ==' ' or res_id =='H_MSE' or res_id =='H_M3L' or res_id=='H_CAS') and is_regular_res:
					ires = ires + 1
					res_name = res.get_resname()
					residue_no = res.get_id()[1]
					for atom in res:
						iatom = iatom + 1
						atom_name = atom.get_name()
						xyz = atom.get_coord()

#						residue_no = atom.get_full_id()[3][1]
						atoms.append( Atom(iatom, atom_name, residue_no, res_name, xyz) )

	out = open(output, 'w')
	out.write(" Structure-Based gro file\n")
	out.write( ("            "+str(len(atoms)))[-12:] )
	out.write("\n")
	for iatom in atoms:
		iatom.write_(out)
	out.close()
