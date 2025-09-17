try:
    from openmm import CustomCompoundBondForce, CustomBondForce, CustomCVForce, CustomCentroidBondForce, CustomExternalForce
    from openmm.unit import kilojoule_per_mole, nanometers, kilocalorie_per_mole, Quantity
except ModuleNotFoundError:
    from simtk.openmm import CustomCompoundBondForce, CustomBondForce, CustomCVForce, CustomCentroidBondForce, CustomExternalForce
    from simtk.unit import kilojoule_per_mole, nanometers, kilocalorie_per_mole, Quantity
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from typing import List, Optional, Union, Tuple, Dict, Any
from openawsem.openAWSEM import OpenMMAWSEMSystem


def read_reference_structure_for_q_calculation_3(oa: OpenMMAWSEMSystem, 
                                                 pdb_file: str, 
                                                 reference_chain_name: str = "ALL", 
                                                 min_seq_sep: int = 3, 
                                                 max_seq_sep: float = np.inf, 
                                                 contact_threshold: Quantity = 0.95*nanometers, 
                                                 Qflag: int = 0, 
                                                 a: float = 0.1, 
                                                 removeDNAchains: bool = True
                                                 ) -> list:
    """
    Reads the reference structure for Q calculation, considering only the specified chains and sequence separation.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        pdb_file: Path to the PDB file.
        reference_chain_name: Name of the chain(s) to be used for reference. Default is "ALL".
        min_seq_sep: Minimum sequence separation for contacts. Default is 3.
        max_seq_sep: Maximum sequence separation for contacts. Default is infinity.
        contact_threshold: Distance threshold for defining a contact. Default is 0.95 nanometers.
        Qflag: Flag to determine the type of Q calculation. Qw calculation is 0; Qo is 1. Default is 0.
        a: Scaling factor for contact distance. Default is 0.1.
        removeDNAchains: Flag to determine if DNA chains should be removed. Default is True.

    Returns:
        A list of structure interactions.
    """
    structure_interactions = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_file)
    model = structure[0]
    chain_start = 0
    count = 0
    proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
    proteinResidues += ["NGP", "IGL", "IPR"]
    rnaResidues = ['A', 'G', 'C', 'U', 'I']
    dnaResidues = ['DA', 'DG', 'DC', 'DT', 'DI']

    for chain in model.get_chains():
        chain_start += count
        count = 0
        if removeDNAchains and np.all([a.get_resname().strip() in dnaResidues for a in chain.get_residues()]):
            print(f"chain {chain.id} is a DNA chain. will be ignored for Q evaluation")
            continue
        elif removeDNAchains and np.all([a.get_resname().strip() not in proteinResidues for a in chain.get_residues()]):
            print(f"chain {chain.id} is a ligand chain. will be ignored for Q evaluation")
            continue
        # print(chain)
        for i, residue_i in enumerate(chain.get_residues()):
            #  print(i, residue_i)
            count +=1
            for j, residue_j in enumerate(chain.get_residues()): # compare each residue to earch other residue
                if abs(i-j) >= min_seq_sep and abs(i-j) <= max_seq_sep:  # taking the signed value to avoid double counting
                    ca_i = residue_i['CA']

                    ca_j = residue_j['CA']

                    r_ijN = abs(ca_i - ca_j)/10.0*nanometers # measurre distance and convert to nm
                    if Qflag ==1 and r_ijN >= contact_threshold: continue
                    sigma_ij = a*(abs(i-j)**0.15)  # scaling factor a times the distance in sequence space ** 0.15
                    gamma_ij = 1.0

                    if reference_chain_name != "ALL" and (chain.id not in reference_chain_name):
                        continue
                    i_index = oa.ca[i+chain_start]
                    j_index = oa.ca[j+chain_start]
                    structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                    structure_interactions.append(structure_interaction)
    # print("Done reading")
    # print(structure_interactions)
    return structure_interactions


def read_reference_structure_for_qc_calculation(oa: OpenMMAWSEMSystem, 
                                                pdb_file: str, 
                                                min_seq_sep: int = 3, 
                                                a: float = 0.1, 
                                                startResidueIndex: int = 0, 
                                                endResidueIndex: int = -1, 
                                                residueIndexGroup: Optional[List[int]] = None
                                                ) -> List[List[Union[int, List[float]]]]:
    """Reads the reference structure for Qc calculation from a pdb file.

    This function uses the canonical Qw/Qo calculation for reference Q.
    For Qw calculation, Qflag is 0; for Qo, Qflag is 1.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        pdb_file: The path to the pdb file.
        min_seq_sep: The minimum sequence separation for considering a contact. Default is 3.
        a: The scaling factor for the contact distance. Default is 0.1.
        startResidueIndex: The index of the first residue to consider in the calculation. Default is 0.
        endResidueIndex: The index of the last residue to consider in the calculation. Default is -1, which means the last residue in the structure.
        residueIndexGroup: A list of residue indices to consider for the calculation. 
        If None, all residues are considered. Default is None.

    Returns:
        A list of lists containing the indices of the residues in contact and the parameters for the contact potential.
    """
    structure_interactions = []
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    if endResidueIndex == -1:
        endResidueIndex = len(list(structure.get_residues()))
    if residueIndexGroup is None:
        # residueIndexGroup is used for non-continuous residues that used for Q computation.
        residueIndexGroup = list(range(len(list(structure.get_residues()))))
    for i, res_i in enumerate(structure.get_residues()):
        chain_i = res_i.get_parent().id
        if i < startResidueIndex:
            continue
        if i not in residueIndexGroup:
            continue
        for j, res_j in enumerate(structure.get_residues()):
            if j > endResidueIndex:
                continue
            if j not in residueIndexGroup:
                continue
            chain_j = res_j.get_parent().id
            if j-i >= min_seq_sep and (chain_i == chain_j).all():
                    ca_i = res_i['CA']
                    ca_j = res_j['CA']

                    r_ijN = abs(ca_i - ca_j)/10.0  # convert to nm
                    sigma_ij = a*(abs(i-j)**0.15)  # 0.1 nm = 1 A
                    gamma_ij = 1.0
                    i_index = oa.ca[i]
                    j_index = oa.ca[j]
                    structure_interaction = [i_index, j_index, [gamma_ij, r_ijN, sigma_ij]]
                    structure_interactions.append(structure_interaction)
    return structure_interactions


def q_value(oa: OpenMMAWSEMSystem, reference_pdb_file: str, 
            reference_chain_name: str = "ALL", 
            min_seq_sep: int = 3, 
            max_seq_sep: float = np.inf, 
            contact_threshold: Quantity = 0.95*nanometers, 
            forceGroup: int = 1
            ) -> CustomBondForce:
    """
    Calculate the Q value (native contacts) of the current structure compared to a reference structure.

    This function computes the canonical QW/QO, which is a measure of the similarity between the current structure and a reference structure based on the number of native contacts.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        reference_pdb_file: The path to the pdb file of the reference structure.
        reference_chain_name: The chain identifier in the reference structure to be considered. Default is "ALL".
        min_seq_sep: The minimum sequence separation for considering a contact. Default is 3.
        max_seq_sep: The maximum sequence separation for considering a contact. Default is infinity (np.inf).
        contact_threshold: The distance below which two residues are considered to be in contact. Default is 0.95 nanometers.
        forceGroup: The force group to which the computed Q value should be added. Default is 1.

    Returns:
        A CustomBondForce object representing the Q value.
    """
    ### Modified by Mingchen to compute canonical QW/QO

    # create bonds
    # structure_interactions = oa.read_reference_structure_for_q_calculation(reference_pdb_file, reference_chain_name, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep, contact_threshold=contact_threshold)
    structure_interactions = read_reference_structure_for_q_calculation_3(oa, reference_pdb_file, reference_chain_name=reference_chain_name,
        min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep, contact_threshold=contact_threshold, Qflag=0)
    # print(len(structure_interactions))
    # print(structure_interactions)
    # create bond force for q calculation
    normalization = len(structure_interactions)
    qvalue = CustomBondForce(f"(1/{normalization})*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    qvalue.addPerBondParameter("gamma_ij")
    qvalue.addPerBondParameter("r_ijN")
    qvalue.addPerBondParameter("sigma_ij")

    for structure_interaction in structure_interactions:
        qvalue.addBond(*structure_interaction)
    qvalue.setForceGroup(forceGroup)
    return qvalue


def qc_value(oa: OpenMMAWSEMSystem, 
             reference_pdb_file: str, 
             min_seq_sep: int = 10, 
             a: float = 0.2
             ) -> CustomBondForce:
    """Calculate the Qc value (non-native contacts) of the current structure compared to a reference structure.

    This function computes the Qc value, which is a measure of the non-native contacts between the current structure and a reference structure.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        reference_pdb_file: The path to the pdb file of the reference structure.
        min_seq_sep: The minimum sequence separation for considering a contact. Default is 10.
        a: The scaling factor for the interaction strength. Default is 0.2.

    Returns:
        A CustomBondForce object representing the Qc value.
    """
    # create bonds
    # structure_interactions = oa.read_reference_structure_for_q_calculation(reference_pdb_file, reference_chain_name, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep, contact_threshold=contact_threshold)
    structure_interactions = read_reference_structure_for_qc_calculation(oa, reference_pdb_file, min_seq_sep=min_seq_sep, a=a)
    # print(len(structure_interactions))
    # print(structure_interactions)

    normalization = len(structure_interactions)
    qvalue = CustomBondForce(f"(1/{normalization})*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    qvalue.addPerBondParameter("gamma_ij")
    qvalue.addPerBondParameter("r_ijN")
    qvalue.addPerBondParameter("sigma_ij")
    for structure_interaction in structure_interactions:
        qvalue.addBond(*structure_interaction)
    qvalue.setForceGroup(3)
    return qvalue


def partial_q_value(oa: OpenMMAWSEMSystem, 
                    reference_pdb_file: str, 
                    min_seq_sep: int = 3, 
                    a: float = 0.1, 
                    startResidueIndex: int = 0, 
                    endResidueIndex: int = -1, 
                    residueIndexGroup: Optional[List[int]] = None, 
                    forceGroup: int = 4
                    ) -> CustomBondForce:
    """Computes the partial Q value for a segment of the protein structure.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        reference_pdb_file: The path to the pdb file of the reference structure.
        min_seq_sep: The minimum sequence separation for considering a contact. Default is 3.
        a: The scaling factor for the interaction strength. Default is 0.1.
        startResidueIndex: The index of the first residue in the segment. Default is 0.
        endResidueIndex: The index of the last residue in the segment. Default is -1, which indicates the last residue of the protein.
        residueIndexGroup: A list of residue indices to be included in the computation. Default is None, which includes all residues.
        forceGroup: The force group to which this interaction should be added. Default is 4.

    Returns:
        A CustomBondForce object representing the partial Q value.
    """
    print(f"Including partial q value computation, start residue index: {startResidueIndex}, end residue index: {endResidueIndex}, residueIndexGroup: {residueIndexGroup}")
    # create bonds
    # structure_interactions = oa.read_reference_structure_for_q_calculation(reference_pdb_file, reference_chain_name, min_seq_sep=min_seq_sep, max_seq_sep=max_seq_sep, contact_threshold=contact_threshold)
    structure_interactions = read_reference_structure_for_qc_calculation(oa, reference_pdb_file, min_seq_sep=min_seq_sep, a=a, startResidueIndex=startResidueIndex, endResidueIndex=endResidueIndex, residueIndexGroup=residueIndexGroup)
    # print(len(structure_interactions))
    # print(structure_interactions)
    if len(structure_interactions) == 0:
        print("No atom found, Please check your startResidueIndex and endResidueIndex.")
        exit()
    normalization = len(structure_interactions)
    qvalue = CustomBondForce(f"(1/{normalization})*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    qvalue.addPerBondParameter("gamma_ij")
    qvalue.addPerBondParameter("r_ijN")
    qvalue.addPerBondParameter("sigma_ij")
    for structure_interaction in structure_interactions:
        qvalue.addBond(*structure_interaction)
    qvalue.setForceGroup(forceGroup)
    return qvalue


def qbias_term(oa: OpenMMAWSEMSystem, 
               q0: float, 
               reference_pdb_file: str, 
               reference_chain_name: str, 
               k_qbias: Quantity = 100*kilocalorie_per_mole, 
               qbias_min_seq_sep: int = 3, 
               qbias_max_seq_sep: float = np.inf, 
               qbias_contact_threshold: Quantity = 0.8*nanometers, 
               forceGroup: int = 4) -> CustomCVForce:
    """Calculates the biasing force based on the deviation of the current Q value from a target Q value.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        q0: The target Q value.
        reference_pdb_file: The path to the pdb file of the reference structure.
        reference_chain_name: The name of the chain in the reference pdb file.
        k_qbias: The force constant for the biasing force. Default is 100*kilocalorie_per_mole.
        qbias_min_seq_sep: The minimum sequence separation for considering a contact. Default is 3.
        qbias_max_seq_sep: The maximum sequence separation for considering a contact. Default is np.inf.
        qbias_contact_threshold: The distance threshold to consider as a contact. Default is 0.8*nanometers.
        forceGroup: The force group to which this interaction should be added. Default is 4.

    Returns:
        A CustomCVForce object representing the biasing force.
    """
    k_qbias = k_qbias.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    qbias = CustomCVForce(f"0.5*{k_qbias}*(q-{q0})^2")
    q = q_value(oa, reference_pdb_file, reference_chain_name, min_seq_sep=qbias_min_seq_sep, max_seq_sep=qbias_max_seq_sep, contact_threshold=qbias_contact_threshold)
    qbias.addCollectiveVariable("q", q)
    # qbias.addGlobalParameter("k_qbias", k_qbias)
    # qbias.addGlobalParameter("q0", q0)
    qbias.setForceGroup(forceGroup)
    return qbias


def create_dist(oa: OpenMMAWSEMSystem, 
                fileA: str = "groupA.dat", 
                fileB: str = "groupB.dat", 
                forceGroup: int = 4
                ) -> CustomCentroidBondForce:
    """Create a CustomCentroidBondForce object that calculates the distance between the centroids of two groups of atoms.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        fileA: The filename containing the indices of the first group of atoms. Defaults to "groupA.dat".
        fileB: The filename containing the indices of the second group of atoms. Defaults to "groupB.dat".
        forceGroup: The force group to which this interaction should be added. Defaults to 4.

    Returns:
        A CustomCentroidBondForce object that computes the distance between the centroids of group A and group B.
    """
    #groupA = list(range(68))
    #groupB = list(range(196, oa.natoms))
    #print(cA)
    #print(cB)
    groupA = np.array(np.loadtxt(fileA,dtype=int)).tolist()
    #print (groupA)
    #groupA = [0,1]
    groupB = np.array(np.loadtxt(fileB,dtype=int)).tolist()
    #groupA, groupB = get_contact_atoms('crystal_structure-openmmawsem.pdb', chainA=cA, chainB=cB)
    #pull_d = CustomCentroidBondForce(2, 'distance(g1,g2)-R0') # 为什么这里给了R0，实际distance变成了2倍？
    #pull_d.addGlobalParameter("R0", 0.0*angstroms)
    pull_d = CustomCentroidBondForce(2, 'distance(g1,g2)')
    pull_d.addGroup(groupA)
    pull_d.addGroup(groupB) # addGroup(groupB)
    pull_d.addBond([0, 1])
    pull_d.setForceGroup(forceGroup)
    return pull_d


def create_centroid_system(oa: OpenMMAWSEMSystem, 
                           fileA: str = "groupA.dat", 
                           fileB: str = "groupB.dat", 
                           k: float = 100.0, 
                           R0: float = 0.0, 
                           forceGroup: int = 26
                           ) -> CustomCVForce:
    """Create a centroid-based system for applying a harmonic restraint between two groups of atoms.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        fileA: The filename containing the indices of the first group of atoms. Defaults to "groupA.dat".
        fileB: The filename containing the indices of the second group of atoms. Defaults to "groupB.dat".
        k: The harmonic force constant. Defaults to 100.0.
        R0: The equilibrium distance between the centroids of the two groups. Defaults to 0.0.
        forceGroup: The force group to which this interaction should be added. Defaults to 26.

    Returns:
        A CustomCVForce object that applies a harmonic restraint between the centroids of two groups of atoms.
    """
    K_pull = k*4.184 * oa.k_awsem
    #R0 = R0*nm
    pull_force = CustomCVForce("0.5*K_pull*(d-R0)^2")
    d = create_dist(oa)#    cA = list(range(68)),
    pull_force.addCollectiveVariable("d", d)
    pull_force.addGlobalParameter("K_pull", K_pull)
    pull_force.addGlobalParameter("R0", R0)
    pull_force.setForceGroup(forceGroup)
    return pull_force


def create_dist_vector(oa: OpenMMAWSEMSystem, 
                       fileA: str = "groupA.dat", 
                       fileB: str = "groupB.dat", 
                       fileC: str = "groupC.dat", 
                       forceGroup: int = 4
                       ) -> CustomCentroidBondForce:
    """Create a distance vector between three groups of atoms.

    Args:
        oa: An OpenMMAWSEMSystem which contains information about the system.
        fileA: The filename containing the indices of the first group of atoms. Defaults to "groupA.dat".
        fileB: The filename containing the indices of the second group of atoms. Defaults to "groupB.dat".
        fileC: The filename containing the indices of the third group of atoms. Defaults to "groupC.dat".
        forceGroup: The force group to which this interaction should be added. Defaults to 4.

    Returns:
        A CustomCentroidBondForce object that computes a vector based on the positions of the three groups of atoms.
    """
    #groupA = list(range(68))
    #groupB = list(range(196, oa.natoms))
    #print(cA)
    #print(cB)
    groupA = np.array(np.loadtxt(fileA,dtype=int)).tolist()
    groupB = np.array(np.loadtxt(fileB,dtype=int)).tolist()
    groupC = np.array(np.loadtxt(fileC,dtype=int)).tolist()
    print (groupA)
    print (groupB)
    print (groupC)
    #groupA, groupB = get_contact_atoms('crystal_structure-openmmawsem.pdb', chainA=cA, chainB=cB)
    #pull_d = CustomCentroidBondForce(2, 'distance(g1,g2)-R0') # 为什么这里给了R0，实际distance变成了2倍？
    #pull_d.addGlobalParameter("R0", 0.0*angstroms)
    pull_d = CustomCentroidBondForce(3, f"r1*cos(theta);\
                                r1=distance(p1,p2);\
                                theta=angle(p1, p2, p3);")

    #pull_d = CustomCompoundBondForce(3, f"r1*cos(theta);\
     #                           r1=distance(p1,p2);\
      #                          theta=angle(p1, p2, p3);")

    g1=pull_d.addGroup(groupA)
    g2=pull_d.addGroup(groupB) # addGroup(groupB)
    #pull_d.addBond([0,1,2])
    g3=pull_d.addGroup(groupC)
    #g4=pull_d.addGroup(groupA)
    #g5=pull_d.addGroup(groupB)


    #print (g1,g2,g3,g4)
    #pull_d.addBond([2800,3435,3780])
    pull_d.addBond([0,1,2],[])
    #pull_d.addBond([g0])
    #pull_d.addBond([1, 2])
    #pull_d.addBond([0, 3])
    pull_d.setForceGroup(forceGroup)
    return pull_d


def create_centroid_system2(oa: OpenMMAWSEMSystem, 
                            fileA: str = "groupA.dat", 
                            fileB: str = "groupB.dat", 
                            fileC: str = "groupC.dat", 
                            k: float = 100.0, 
                            R0: float = 0.0, 
                            forceGroup: int = 26
                            ) -> CustomCVForce:
    """Create a centroid-based system with a harmonic bias potential.

    Args:
        oa: An instance of OpenMMAWSEMSystem representing the system.
        fileA: Filename containing the indices of the first group of atoms. Defaults to "groupA.dat".
        fileB: Filename containing the indices of the second group of atoms. Defaults to "groupB.dat".
        fileC: Filename containing the indices of the third group of atoms. Defaults to "groupC.dat".
        k: The harmonic force constant. Defaults to 100.
        R0: The equilibrium distance for the harmonic potential. Defaults to 0.
        forceGroup: The force group to which this interaction should be added. Defaults to 26.

    Returns:
        A CustomCVForce object with the harmonic bias potential applied to the centroid of the three groups of atoms.
    """
    K_pull = k*4.184 * oa.k_awsem
    #R0 = R0*nm
    pull_force = CustomCVForce("0.5*K_pull*(d-R0)^2")
    d = create_dist_vector(oa,fileA=fileA,fileB=fileB,fileC=fileC)#    cA = list(range(68)),
    pull_force.addCollectiveVariable("d", d)
    pull_force.addGlobalParameter("K_pull", K_pull)
    pull_force.addGlobalParameter("R0", R0)
    pull_force.setForceGroup(forceGroup)
    return pull_force


def rg_term(oa: OpenMMAWSEMSystem, 
            convertToAngstrom: bool = True
            ) -> CustomCVForce:
    """Calculate the radius of gyration term for the given OpenMMAWSEMSystem.

    Args:
        oa: An instance of OpenMMAWSEMSystem representing the system.
        convertToAngstrom: A boolean indicating whether to convert the result to Angstroms. Defaults to True.

    Returns:
        A CustomCVForce object representing the radius of gyration term.
    """
    rg_square = CustomBondForce("1/normalization*r^2")
    # rg = CustomBondForce("1")
    rg_square.addGlobalParameter("normalization", oa.nres*oa.nres) # add the normalization factor as the sqared number of resiues 
    for i in range(oa.nres): # for each residue
        for j in range(i+1, oa.nres): # get each other residue in the structure
            rg_square.addBond(oa.ca[i], oa.ca[j], []) # acd a bond between them
    if convertToAngstrom:
        unit = 10
    else:
        unit = 1
    rg = CustomCVForce(f"{unit}*rg_square^0.5") # caclulate the radius of gyration
    rg.addCollectiveVariable("rg_square", rg_square) # add the r gyr as a CV
    rg.setForceGroup(2)
    return rg


def rg_bias_term(oa: OpenMMAWSEMSystem, 
                 k: Quantity = 1*kilocalorie_per_mole, 
                 rg0: float = 0.0, 
                 atomGroup: Union[int, List[int]] = -1, 
                 forceGroup: int = 27
                 ) -> CustomCVForce:
    """Apply a radius of gyration bias to a group of atoms in the OpenMMAWSEMSystem.

    Args:
        oa: An instance of OpenMMAWSEMSystem representing the system.
        k: The harmonic force constant. Defaults to 1 kilocalorie_per_mole.
        rg0: The reference (equilibrium) radius of gyration. Defaults to 0.
        atomGroup: The indices of the atoms to which the bias will be applied. 
        If set to -1, the bias is applied to all atoms. Defaults to -1.
        forceGroup: The force group to which this interaction should be added. Defaults to 27.

    Returns:
        A CustomCVForce object with the radius of gyration bias applied.
    """
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_rg = oa.k_awsem * k
    nres, ca = oa.nres, oa.ca # get the indeces of residues and C-alpha atoms
    if atomGroup == -1:
        group = list(range(nres))
    else:
        group = atomGroup     # atomGroup = [0, 1, 10, 12]  means include residue 1, 2, 11, 13.
    n = len(group) # the number of residues
    normalization = n*n # the normalization factor
    rg_square = CustomBondForce(f"1.0/{normalization}*r^2")
    # rg = CustomBondForce("1")          
    # rg_square.addGlobalParameter("normalization", n*n)
    for i in group:
        for j in group:
            if j <= i:
                continue
            rg_square.addBond(ca[i], ca[j], [])
    rg = CustomCVForce(f"{k_rg}*(rg_square^0.5-{rg0})^2")
    rg.addCollectiveVariable("rg_square", rg_square)
    rg.setForceGroup(forceGroup)
    return rg


def cylindrical_rg_bias_term(oa: OpenMMAWSEMSystem, 
                             k: Quantity = 1*kilocalorie_per_mole, 
                             rg0: float = 0.0, 
                             atomGroup: Union[int, List[int]] = -1, 
                             forceGroup: int = 27
                             ) -> CustomCVForce:
    """Calculate a cylindrical radius of gyration bias term for a group of atoms.

    This function applies a cylindrical radius of gyration bias to a group of atoms in the OpenMMAWSEMSystem.

    Args:
        oa: An instance of OpenMMAWSEMSystem representing the system.
        k: The harmonic force constant. Defaults to 1 kilocalorie_per_mole.
        rg0: The reference (equilibrium) radius of gyration. Defaults to 0.
        atomGroup: The indices of the atoms to which the bias will be applied. If set to -1, the bias is applied to all atoms. Defaults to -1.
        forceGroup: The force group to which this interaction should be added. Defaults to 27.

    Returns:
        A CustomCVForce object with the cylindrical radius of gyration bias applied.
    """
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_rg = oa.k_awsem * k
    nres, ca = oa.nres, oa.ca
    if atomGroup == -1:
        group = list(range(nres))
    else:
        group = atomGroup          # atomGroup = [0, 1, 10, 12]  means include residue 1, 2, 11, 13.
    n = len(group)
    normalization = n * n
    rg_square = CustomCompoundBondForce(2, f"1/{normalization}*((x1-x2)^2+(y1-y2)^2)")

    for i in group:
        for j in group:
            if j <= i:
                continue
            rg_square.addBond([ca[i], ca[j]], [])

    rg = CustomCVForce(f"{k_rg}*(rg_square^0.5-{rg0})^2")
    rg.addCollectiveVariable("rg_square", rg_square)
    rg.setForceGroup(forceGroup)
    return rg


def pulling_term(oa: OpenMMAWSEMSystem, 
                 k_pulling: float = 4.184, 
                 forceDirect: str = "x", 
                 appliedToResidue: Union[int, str] = 1, 
                 forceGroup: int = 19
                 ) -> CustomExternalForce:
    """Apply a directional pulling force to a specified residue.

    This function adds a constant force in a specified direction to a single residue within the system.

    Args:
        oa: An instance of OpenMMAWSEMSystem representing the system.
        k_pulling: The magnitude of the pulling force in units of kJ/nm. Defaults to 4.184, which is equivalent to 1 kcal/nm.
        forceDirect: The direction of the pulling force. Can be 'x', 'y', or 'z'. Defaults to 'x'.
        appliedToResidue: The index of the residue to which the force is applied. Can be an integer index, 'FIRST' for the first residue, or 'LAST' for the last residue. Defaults to 1.
        forceGroup: The force group to which this interaction should be added. Defaults to 19.

    Returns:
        A CustomExternalForce object with the pulling force applied to the specified residue.
    """
    k_pulling *= oa.k_awsem
    pulling = CustomExternalForce(f"(-{k_pulling})*({forceDirect})")
    for i in range(oa.natoms):
        if appliedToResidue == "LAST":
            appliedToResidue = oa.nres
        if appliedToResidue == "FIRST":
            appliedToResidue = 1
        if oa.resi[i] == (appliedToResidue-1):
            pulling.addParticle(i)
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
    pulling.setForceGroup(forceGroup)
    return pulling
