#!/usr/bin/env python3
import os
import argparse
import sys
import openawsem
import pandas as pd
from pathlib import Path
import contextlib
import subprocess
from typing import Optional, Tuple, Union
import shutil


__location__ = openawsem.__location__
__author__ = 'Wei Lu'


def parse_arguments():
    """
    This function defines and parses command-line arguments for a protein simulation project template creation script.
    It uses the argparse module to define the expected arguments and their corresponding help messages.
    """
    
    # Create an argument parser with a description for the script
    parser = argparse.ArgumentParser(
        description=f"This Python 3 script automatically creates a protein simulation project template quickly and efficiently. Written by {__author__}"
    )

    # Define the expected arguments and their help messages
    parser.add_argument("proteins", nargs="*", help="Provide the names of the proteins (e.g., 1r69) or the target PDB files for the simulation, separated by spaces.")
    parser.add_argument("-c", "--chain", default="-1", help="Specify the chains to be simulated (e.g., 'ABC').")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--frag", action="store_true", default=False, help="Generate fragment memories.")
    parser.add_argument("--extended", action="store_true", default=False, help="Start from an extended structure generated using PyMOL (ensure it's installed). Supports single chain only.")
    parser.add_argument("--membrane", action="store_true", default=False, help="Enable membrane protein simulations.")
    parser.add_argument("--hybrid", action="store_true", default=False, help="Enable hybrid simulations.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output.")
    # parser.add_argument("--predict_ssweight_from_fasta", action="store_true", default=False, help="Predict secondary structure weight from FASTA sequence.")
    parser.add_argument("--keepIds", action="store_true", default=False, help="Preserve chain and residue index. By default, chains will be renamed from 'A' and indices will start from 1.")
    parser.add_argument("--keepLigands", action="store_true", default=False, help="Preserve ligands in the protein structure.")

      # Create a subparser for frag-related arguments
    frag_parser = parser.add_argument_group("frag", "Arguments for fragment memory generation. Only used if --frag is specified")
    frag_parser.add_argument("--frag_database", default=openawsem.data_path.blast, help="Specify the database for fragment generation.")
    frag_parser.add_argument("--frag_fasta", default=None, help="Provide the FASTA file for fragment generation.")
    frag_parser.add_argument("--frag_N_mem", type=int, default=20, help="Number of memories to generate per fragment.")
    frag_parser.add_argument("--frag_brain_damage", type=float, choices=[0, 0.5, 1, 2], default=0, help="Control the inclusion or exclusion of homologous protein structures for generating fragment memories.\n 0: include all homologs, 0.5: include only self-structures, 1: exclude all homologs, 2: include only non-homologous structures.")
    frag_parser.add_argument("--frag_fragmentLength", type=int, default=9, help="SLength of the fragments to be generated.")


    # Parse and return the command-line arguments
    return parser.parse_args()


class AWSEMSimulationProject:
    def __init__(self, args: argparse.Namespace):
        """Initialize the AWSEMSimulationProject with command line arguments.

        Args:
            args (argparse.Namespace): Command line arguments parsed by argparse.
        """
        self.data_path = __location__
        self.base_folder = Path.cwd()  # Project folder
        self.args = args
                
    def run_command(self, command: list, stdout: Optional[str] = None):
        """Run a shell command with optional stdout redirection.

        Args:
            command (list): The command to run as a list of strings.
            stdout (Optional[str]): The file to write stdout to. Defaults to None.
        """
        if self.args.debug:
            print(' '.join(command))
        else:
            subprocess.run(command, check=True, shell=False, stdout=(open(stdout, "w") if stdout else None))

    @contextlib.contextmanager
    def change_directory(self, path: Union[str, Path]):
        """Change the current working directory to the given path within a context.

        Args:
            path (Union[str, Path]): The path to change the working directory to.

        Yields:
            None
        """
        old_path = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_path)

    def log_commandline_args(self, log_file: str = 'create_project_commandline_args.txt'):
        """Log the command line arguments used to run the script to a file.

        Args:
            log_file (str): The file to write the command line arguments to. Defaults to 'create_project_commandline_args.txt'.
        """
        with open(log_file, 'w') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')

    def prepare_input_files_from_pdb(self, parent_folder: Union[str, Path] = ".") -> Tuple[str, str]:
        """Prepare input files from a PDB file.

        Args:
            parent_folder (Union[str, Path]): The parent directory where the 'original_pdbs' directory will be created. Defaults to ".".

        Returns:
            Tuple[str, str]: A tuple containing the protein name without the file extension and the PDB filename.
        """
        protein_path = Path(self.args.protein)
        parent_folder = Path(parent_folder)
        assert protein_path.exists(), f"The protein path {str(protein_path)} does not exist"
        
        name = protein_path.stem
        pdb = protein_path.name
        
        original_pdbs_dir = parent_folder / "original_pdbs"
        original_pdbs_dir.mkdir(parents=True, exist_ok=True)
        (original_pdbs_dir / pdb).write_bytes(protein_path.read_bytes())
        
        return name, pdb

    def prepare_input_files_from_fasta(self, parent_folder: Union[str, Path] = ".") -> Tuple[str, str]:
        """Prepare input files from a FASTA file.

        Args:
            parent_folder (Union[str, Path]): The parent directory where the 'original_fasta' and 'original_pdbs' directories will be created. Defaults to ".".

        Returns:
            Tuple[str, str]: A tuple containing the protein name without the file extension and the PDB filename.
        """
        print("Creating simulation folder from FASTA file.")
        protein_path = Path(self.args.protein)
        name = protein_path.stem

        try:
            self.run_command(["python3", __location__/"helperFunctions"/"fasta2pdb.py", name, "-f", self.args.protein])
            openawsem.helperFunctions.add_chain_to_pymol_pdb(f"{name}.pdb")
        except Exception as e:
            print(f"ERROR: Failed to convert FASTA to PDB. Exception: {e}")
            exit()

        original_fasta_dir = Path(parent_folder) / "original_fasta"
        original_pdbs_dir = Path(parent_folder) / "original_pdbs"
        original_fasta_dir.mkdir(parents=True, exist_ok=True)
        original_pdbs_dir.mkdir(parents=True, exist_ok=True)
        fasta_file_path = (original_fasta_dir / protein_path.name)
        fasta_file_path.write_bytes(protein_path.read_bytes())
        pdb_path = Path(f"{name}.pdb")
        crystal_structure_path = (Path(parent_folder) / "crystal_structure.pdb")
        crystal_structure_path.write_bytes(pdb_path.read_bytes())
        pdb = pdb_path.name
        
        return name, pdb

    def prepare_input_files_from_name(self, parent_folder: Union[str, Path] = ".") -> Tuple[str, str]:
        """Prepare input files from a protein name.

        Args:
            parent_folder (Union[str, Path]): The parent directory where the 'original_pdbs' directory will be created. Defaults to ".".

        Returns:
            Tuple[str, str]: A tuple containing the protein name without the file extension and the PDB filename.
        """
        name = self.args.protein
        pdb = f"{name}.pdb"
        pdb_list = [name]
        parent_folder = Path(parent_folder)
        
        try:
            openawsem.helperFunctions.downloadPdb(pdb_list, location=parent_folder/'original_pdbs')
        except Exception as e:
            print(f"ERROR: Failed to download PDB file. Exception: {e}")
            exit()

        return name, pdb

    def process_pdb_files(self):
        """Process the PDB files by cleaning, preparing, and generating additional required files."""
        removeHeterogens = not self.args.keepLigands
        chain = self.args.chain

        if not Path("crystal_structure.pdb").exists():
            openawsem.helperFunctions.cleanPdb(
                [self.name],
                chain=chain,
                toFolder="cleaned_pdbs",
                verbose=self.args.verbose,
                keepIds=True,
                removeHeterogens=removeHeterogens
            )
            cleaned_pdb_path = Path(f"cleaned_pdbs/{self.pdb}")
            shutil.copy(cleaned_pdb_path, "crystal_structure.pdb")

        if chain == "-1":
            chain = openawsem.helperFunctions.getAllChains(
                "crystal_structure.pdb",
                removeDNAchains=True
            )
            print("Chains info read from crystal_structure.pdb, chains to simulate: ", chain)

        input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb(
            "crystal_structure.pdb",
            chain,
            use_cis_proline=False,
            keepIds=self.args.keepIds,
            removeHeterogens=removeHeterogens
        )
        openawsem.ensure_atom_order(input_pdb_filename)

        self.input_pdb_filename = input_pdb_filename
        self.cleaned_pdb_filename = cleaned_pdb_filename
        
        self.chain = openawsem.helperFunctions.getAllChains("crystal_structure-cleaned.pdb")
        openawsem.getSeqFromCleanPdb(input_pdb_filename, chains=chain, writeFastaFile=True)
        shutil.copy('crystal_structure.fasta', f'{self.name}.fasta')
        
        if self.args.extended:
            openawsem.helperFunctions.create_extended_pdb_from_fasta(f"{self.name}.fasta", output_file_name="extended.pdb")
            input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb("extended.pdb", "A", use_cis_proline=False, keepIds=self.args.keepIds, removeHeterogens=removeHeterogens)
            openawsem.ensure_atom_order(input_pdb_filename)
        
        shutil.copy('crystal_structure-cleaned.pdb', f'{self.pdb}')
        
        if self.args.keepLigands:
            shutil.copy("crystal_structure-cleaned.pdb", f"{self.name}-cleaned.pdb")
            with open("tmp.pdb", "w") as output_file:
                with open("crystal_structure-openmmawsem.pdb", "r") as input_file:
                    for line in input_file:
                        if "ATOM" in line:
                            output_file.write(line)
                with open("crystal_structure-cleaned.pdb", "r") as input_file:
                    for line in input_file:
                        if "HETATM" in line:
                            output_file.write(line)
            os.rename("tmp.pdb", f"{self.name}-openmmawsem.pdb")
        else:
            input_pdb_filename, cleaned_pdb_filename = openawsem.prepare_pdb(self.pdb, self.chain, keepIds=self.args.keepIds, removeHeterogens=removeHeterogens)
            openawsem.ensure_atom_order(input_pdb_filename)

    def generate_ssweight_from_stride(self):
        """Generate the secondary structure weight file (ssweight) using stride or Predict_Property."""
        self.run_command(["stride", "crystal_structure.pdb"], stdout="ssweight.stride")
        self.run_command(["python", __location__/"helperFunctions"/"stride2ssweight.py"], stdout="ssweight")
        protein_length = openawsem.helperFunctions.getFromTerminal("wc ssweight").split()[0]
        if int(protein_length) == 0:
            seq = openawsem.helperFunctions.read_fasta(f"{self.name}.fasta")
            protein_length = len(seq)
            print("impose no secondary bias.")
            print("you might want to install Predict_Property and use the predict_ssweight_from_fasta option.")
            with open("ssweight", "w") as out:
                for i in range(protein_length):
                    out.write("0.0 0.0\n")
        print(f"protein: {self.name}, length: {protein_length}")

    def generate_ssweight_from_fasta(self):
        """Generate the secondary structure weight file (ssweight) from a FASTA file."""
        self.run_command(["$Predict_Property/Predict_Property.sh", "-i", f"{self.name}.fasta"])
        from_secondary = f"{self.name}_PROP/{self.name}.ss3"
        toPre = "."
        to_ssweight = f"{toPre}/ssweight"
        print("convert ssweight")
        data = pd.read_csv(from_secondary, comment="#", names=["i", "Res", "ss3", "Helix", "Sheet", "Coil"], sep=r"\s+")
        with open(to_ssweight, "w") as out:
            for i, line in data.iterrows():
                if line["ss3"] == "H":
                    out.write("1.0 0.0\n")
                if line["ss3"] == "E":
                    out.write("0.0 1.0\n")
                if line["ss3"] == "C":
                    out.write("0.0 0.0\n")
                        
    def prepare_membrane_files(self):
        """Prepare the required membrane-related files (zim and zimPosition) if the membrane or hybrid options are specified."""
        self.run_command(["grep", "-E", "CB|CA  GLY", "crystal_structure-cleaned.pdb"], stdout="cbs.data")
        self.run_command(["awk", "{if($9>15) print \"1\"; else if($9<-15) print \"3\"; else print \"2\"}", "cbs.data"], stdout="zimPosition")

        openawsem.helperFunctions.create_zim(f"crystal_structure.fasta", tableLocation=__location__/"helperFunctions")

    def generate_fragment_memory(self, database: str = "cullpdb_pc80_res3.0_R1.0_d160504_chains29712", fasta: Optional[str] = None, N_mem: int = 20, brain_damage: float = 1.0, fragmentLength: int = 9):
        """Generate the fragment memory file if the frag option is specified.

        Args:
            database (str): The database for fragment generation. Defaults to "cullpdb_pc80_res3.0_R1.0_d160504_chains29712".
            fasta (Optional[str]): The FASTA file for fragment generation. Defaults to None.
            N_mem (int): Number of memories to generate per fragment. Defaults to 20.
            brain_damage (float): Control the inclusion or exclusion of homologous protein structures. Defaults to 1.0.
            fragmentLength (int): Length of the fragments to be generated. Defaults to 9.
        """
        if fasta is None:
            fasta = f"{self.name}.fasta"

        openawsem.helperFunctions.create_fragment_memories(database=database, fasta_file=fasta, memories_per_position=N_mem, 
                                                           brain_damage=brain_damage, fragment_length=fragmentLength, pdb_dir=openawsem.data_path.pdb, 
                                                           index_dir=openawsem.data_path.index, frag_lib_dir=openawsem.data_path.gro,
                                                           failed_pdb_list_file=openawsem.data_path.pdbfail, pdb_seqres=openawsem.data_path.pdbseqres,
                                                           weight=1, evalue_threshold=10000, cutoff_identical=90)

        openawsem.helperFunctions.check_and_correct_fragment_memory("frags.mem")

        # Relocate the file to the fraglib folder
        openawsem.helperFunctions.relocate(fileLocation="frags.mem", toLocation="fraglib")

        # Replace the file path in frags.mem
        openawsem.helperFunctions.replace(f"frags.mem", f"{__location__}//Gros/", "./fraglib/")
        self.run_command(["cp", "frags.mem", "frag_memory.mem"])

    def generate_single_memory(self):
        """Generate a single memory file for each chain in the protein."""
        for c in self.chain:
            self.run_command(["python", f"{__location__}/helperFunctions/Pdb2Gro.py", "crystal_structure-cleaned.pdb", f"{self.name}_{c}.gro", f"{c}"])
        
        seq_data = openawsem.helperFunctions.seq_length_from_pdb("crystal_structure-cleaned.pdb", self.chain)
        with open("single_frags.mem", "w") as out:
            out.write("[Target]\nquery\n\n[Memories]\n")
            for (chain_name, chain_start_residue_index, seq_length) in seq_data:
                out.write(f"{self.name}_{chain_name}.gro {chain_start_residue_index} 1 {seq_length} 20\n")

    def generate_charges(self):
        """Generate the charge array for the protein based on its FASTA sequence."""
        openawsem.helperFunctions.generate_charge_array(Path(f"{self.name}.fasta"), Path('charge.txt'))

    def copy_parameters(self, destination_folder: Union[str, Path] = '.'):
        """Copy parameter files to the specified destination folder.

        Args:
            destination_folder (Union[str, Path]): The folder to copy the parameter files to. Defaults to '.'.
        """
        parameter_files = [
            "burial_gamma.dat", "gamma.dat", "membrane_gamma.dat",
            "anti_HB", "anti_NHB", "anti_one", "para_HB", "para_one"
        ]
        for file_name in parameter_files:
            shutil.copy(__location__/"parameters"/file_name, destination_folder)
    
    def copy_scripts(self, destination_folder: Union[str, Path] = '.'):
        """Copy required scripts and files to the specified destination folder.

        Args:
            destination_folder (Union[str, Path]): The folder to copy the scripts to. Defaults to '.'.
        """
        script_files = ["mm_run.py", "mm_analyze.py", "forces_setup.py"]
        for file_name in script_files:
            shutil.copy(__location__/"scripts"/file_name, destination_folder)

    def run(self):
        """Execute the main workflow of the AWSEMSimulationProject class."""
        self.log_commandline_args()

        for protein in self.args.proteins:
            with self.change_directory(self.base_folder):
                self.args.protein = protein
                project_folder = Path(os.path.splitext(os.path.basename(protein))[0])
                project_folder.mkdir(parents=True, exist_ok=True)
                
                if self.args.protein[-4:] == '.pdb':
                    self.name, self.pdb = self.prepare_input_files_from_pdb(project_folder)
                elif self.args.protein[-6:] == ".fasta":
                    self.name, self.pdb = self.prepare_input_files_from_fasta(project_folder)
                else:
                    self.name, self.pdb = self.prepare_input_files_from_name(project_folder)
                
                print(self.name, self.pdb)
                
                with self.change_directory(project_folder):
                    self.process_pdb_files()
                    
                    if self.args.membrane or self.args.hybrid:
                        self.prepare_membrane_files()
                    
                    self.generate_single_memory()

                    if self.args.frag:
                        self.generate_fragment_memory(database=self.args.frag_database, fasta=self.args.frag_fasta, N_mem=self.args.frag_N_mem, brain_damage=self.args.frag_brain_damage, fragmentLength=self.args.frag_fragmentLength)

                    self.generate_charges()
                    
                    self.copy_scripts()

                    self.copy_parameters()

                    print(f"{project_folder} project folder created")
                    print("please modify the forces_setup.py if we want to change what energy terms to be used.")


def main():
    args = parse_arguments()
    project = AWSEMSimulationProject(args)
    project.run()

if __name__=="__main__":
    main()
