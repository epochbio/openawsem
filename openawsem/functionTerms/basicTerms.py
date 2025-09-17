try:
    from openmm import CustomCompoundBondForce, CustomNonbondedForce, HarmonicBondForce, Discrete2DFunction
    from openmm.unit import kilojoule_per_mole, kilocalorie_per_mole, Quantity
except ModuleNotFoundError:
    from simtk.openmm import CustomCompoundBondForce, CustomNonbondedForce, HarmonicBondForce, Discrete2DFunction
    from simtk.unit import kilojoule_per_mole, kilocalorie_per_mole, Quantity
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
from openawsem.openAWSEM import OpenMMAWSEMSystem


one_to_three={'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE',
            'G':'GLY', 'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU',
            'M':'MET', 'N':'ASN', 'P':'PRO', 'Q':'GLN', 'R':'ARG',
            'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'}

three_to_one = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
                'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
                'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
                'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}


def con_term(oa: OpenMMAWSEMSystem, 
             k_con: float = 50208, 
             bond_lengths: List[float] = [.3816, .240, .276, .153], 
             forceGroup: int = 20
             ) -> HarmonicBondForce:
    """
    Add conformational (con) forces to the system as harmonic bond forces.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k_con: The force constant for the conformational term. Default is 50208 kJ/nm^2.
        bond_lengths: A list of equilibrium bond lengths in nanometers. Default is [.3816, .240, .276, .153].
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A HarmonicBondForce object representing the conformational forces.
    """
    # add con forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_con *= oa.k_awsem
    con = HarmonicBondForce()

    if oa.periodic:
        con.setUsesPeriodicBoundaryConditions(True)
    
    for i in range(oa.nres):
        con.addBond(oa.ca[i], oa.o[i], bond_lengths[1], k_con)
        if not oa.res_type[i] == "IGL":  # OpenAWSEM system doesn't have CB for glycine, so the following bond is not exist for Glycine, but LAMMPS include this bond by using virtual HB as CB.
            con.addBond(oa.ca[i], oa.cb[i], bond_lengths[3], k_con)
        if i not in oa.chain_ends:
            con.addBond(oa.ca[i], oa.ca[i+1], bond_lengths[0], k_con)
            con.addBond(oa.o[i], oa.ca[i+1], bond_lengths[2], k_con)
    con.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return con


def chain_term(oa: OpenMMAWSEMSystem, 
               k_chain: float = 50208, 
               bond_k: List[float] = [1, 1, 1], 
               bond_lengths: List[float] = [0.2459108, 0.2519591, 0.2466597], 
               forceGroup: int = 20
               ) -> HarmonicBondForce:
    """Add chain forces to the system.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k_chain: The force constant for the chain term. Default is 50208 kJ/nm^2.
        bond_k: A list of scaling factors for the bond force constants. Default is [1, 1, 1].
        bond_lengths: A list of equilibrium bond lengths in nanometers. Default is [0.2459108, 0.2519591, 0.2466597].
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A HarmonicBondForce object representing the chain forces.
    """
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chain *= oa.k_awsem
    chain = HarmonicBondForce()
    
    if oa.periodic:
        chain.setUsesPeriodicBoundaryConditions(True)

    for i in range(oa.nres):
        if i not in oa.chain_starts and not oa.res_type[i] == "IGL":
            chain.addBond(oa.n[i], oa.cb[i], bond_lengths[0], k_chain*bond_k[0])
        if i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            chain.addBond(oa.c[i], oa.cb[i], bond_lengths[1], k_chain*bond_k[1])
        if i not in oa.chain_starts and i not in oa.chain_ends:
            chain.addBond(oa.n[i], oa.c[i], bond_lengths[2], k_chain*bond_k[2])
    chain.setForceGroup(forceGroup)
    return chain


def chi_term(oa: OpenMMAWSEMSystem, 
             k_chi: float = 251.04, 
             chi0: float = -0.71, 
             forceGroup: int = 20
             ) -> CustomCompoundBondForce:
    """
    Add chi forces to the system.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k_chi: The force constant for the chi angle term. Default is 251.04 kJ/mol.
        chi0: The equilibrium chi angle value in radians. Default is -0.71 radians.
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A CustomCompoundBondForce object representing the chi angle forces.
    """
    # The sign of the equilibrium value is opposite and magnitude differs slightly
    # 251.04 = 60 * 4.184 kJ, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chi *= oa.k_awsem
    chi = CustomCompoundBondForce(4, f"{k_chi}*(chi*norm-{chi0})^2;"
                                        "chi=crossproduct_x*r_cacb_x+crossproduct_y*r_cacb_y+crossproduct_z*r_cacb_z;"
                                        "crossproduct_x=(u_y*v_z-u_z*v_y);"
                                        "crossproduct_y=(u_z*v_x-u_x*v_z);"
                                        "crossproduct_z=(u_x*v_y-u_y*v_x);"
                                        "norm=1/((u_x*u_x+u_y*u_y+u_z*u_z)*(v_x*v_x+v_y*v_y+v_z*v_z)*(r_cacb_x*r_cacb_x+r_cacb_y*r_cacb_y+r_cacb_z*r_cacb_z))^0.5;"
                                        "r_cacb_x=x1-x4;"
                                        "r_cacb_y=y1-y4;"
                                        "r_cacb_z=z1-z4;"
                                        "u_x=x1-x2; u_y=y1-y2; u_z=z1-z2;"
                                        "v_x=x3-x1; v_y=y3-y1; v_z=z3-z1;")

    if oa.periodic:
        chi.setUsesPeriodicBoundaryConditions(True)

    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            chi.addBond([oa.ca[i], oa.c[i], oa.n[i], oa.cb[i]])
    chi.setForceGroup(forceGroup)
    return chi


def excl_term(oa: OpenMMAWSEMSystem, 
              k_excl: float = 8368, 
              r_excl: float = 0.35, 
              periodic: bool = False, 
              excludeCB: bool = False, 
              forceGroup: int = 20
              ) -> CustomNonbondedForce:
    """
    Add excluded volume interactions to the system with an option to exclude interactions between C-beta atoms.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k_excl: The force constant for the excluded volume interactions. Default is 8368 kJ/mol.
        r_excl: The distance within which the excluded volume interaction is active. Default is 0.35 nm.
        periodic: A boolean indicating whether periodic boundary conditions should be used. Default is False.
        excludeCB: A boolean indicating whether interactions between C-beta atoms should be excluded. Default is False.
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A CustomNonbondedForce object representing the excluded volume interactions.
    """
    # add excluded volume
    # Still need to add element specific parameters
    # 8368 = 20 * 4.184 * 100 kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k_excl *= oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step({r_excl}-r)*(r-{r_excl})^2")

    if oa.periodic:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    for i in range(oa.natoms):
        excl.addParticle()
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)
    excl.addInteractionGroup(oa.ca, oa.ca)
    if not excludeCB:
        excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.ca, [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.o, oa.o)

    excl.setCutoffDistance(r_excl)

    # excl.setNonbondedMethod(excl.CutoffNonPeriodic)
    excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    return excl


def excl_term_v2(oa: OpenMMAWSEMSystem, 
                 k_excl: float = 8368, 
                 r_excl: float = 0.35, 
                 periodic: bool = False, 
                 excludeCB: bool = False, 
                 forceGroup: int = 20
                 ) -> CustomNonbondedForce:
    """
    Create a version of the excluded volume term that does not use createExclusionsFromBonds.

    This function sets up a CustomNonbondedForce to model excluded volume interactions in a way that avoids
    potential conflicts with other CustomNonbondedForce objects that could arise from using createExclusionsFromBonds.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k_excl: The force constant for the excluded volume interactions. Default is 8368 kJ/mol.
        r_excl: The distance within which the excluded volume interaction is active. Default is 0.35 nm.
        periodic: A boolean indicating whether periodic boundary conditions should be used. Default is False.
        excludeCB: A boolean indicating whether to exclude interactions between CB atoms. Default is False.
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A CustomNonbondedForce object representing the excluded volume interactions.
    """
    # Implementation goes here
    # add excluded volume
    # Still need to add element specific parameters
    # 8368 = 20 * 4.184 * 100 kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k_excl *= oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step(abs(res1-res2)-2+isChainEdge1*isChainEdge2+isnot_Ca1+isnot_Ca2)*step({r_excl}-r)*(r-{r_excl})^2")
    
    if oa.periodic:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    excl.addPerParticleParameter("res")
    excl.addPerParticleParameter("isChainEdge")
    excl.addPerParticleParameter("isnot_Ca")
    for i in range(oa.natoms):
        # print(oa.resi[i])
        if (i in oa.chain_ends) or (i in oa.chain_starts):
            # print(i)
            isChainEdge = 1
        else:
            isChainEdge = 0
        if (i in oa.ca):
            isnot_Ca = 0
        else:
            isnot_Ca = 1
        excl.addParticle([oa.resi[i], isChainEdge, isnot_Ca])
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)
    excl.addInteractionGroup(oa.ca, oa.ca)
    if not excludeCB:
        excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.ca, [x for x in oa.cb if x > 0])
    excl.addInteractionGroup(oa.o, oa.o)

    excl.setCutoffDistance(r_excl)

    # excl.setNonbondedMethod(excl.CutoffNonPeriodic)
    # print(oa.bonds)
    # print(len(oa.bonds))
    # excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    return excl


def rama_term(oa: OpenMMAWSEMSystem, 
              k_rama: float = 8.368, 
              num_rama_wells: int = 3, 
              w: List[float] = [1.3149, 1.32016, 1.0264], 
              sigma: List[float] = [15.398, 49.0521, 49.0954], 
              omega_phi: List[float] = [0.15, 0.25, 0.65], 
              phi_i: List[float] = [-1.74, -1.265, 1.041], 
              omega_psi: List[float] = [0.65, 0.45, 0.25], 
              psi_i: List[float] = [2.138, -0.318, 0.78], 
              forceGroup: int = 21) -> CustomCompoundBondForce:
    """Add Ramachandran potential to the system.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k_rama: The force constant for the Rama potential. Default is 8.368.
        num_rama_wells: The number of Ramachandran wells. Default is 3.
        w: The weights for the Rama potential. Default is [1.3149, 1.32016, 1.0264].
        sigma: The barrier heights for the Rama potential. Default is [15.398, 49.0521, 49.0954].
        omega_phi: The weights for the phi angle term. Default is [0.15, 0.25, 0.65].
        phi_i: The ideal phi angles. Default is [-1.74, -1.265, 1.041].
        omega_psi: The weights for the psi angle term. Default is [0.65, 0.45, 0.25].
        psi_i: The ideal psi angles. Default is [2.138, -0.318, 0.78].
        forceGroup: The force group to which this force will be added. Default is 21.

    Returns:
        A CustomCompoundBondForce object that implements the Rama potential.
    """
    # add Rama potential
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama *= oa.k_awsem
    rama_function = ''.join(["w%d*exp(-sigma%d*(omega_phi%d*phi_term%d^2+omega_psi%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i) for i in range(num_rama_wells)])[:-1]
    rama_function = '-k_rama*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_wells)])
    rama_string = rama_function+rama_parameters
    rama = CustomCompoundBondForce(5, rama_string)

    if oa.periodic:
        rama.setUsesPeriodicBoundaryConditions(True)

    for i in range(num_rama_wells):
        rama.addGlobalParameter(f"k_rama", k_rama)
        rama.addGlobalParameter(f"w{i}", w[i])
        rama.addGlobalParameter(f"sigma{i}", sigma[i])
        rama.addGlobalParameter(f"omega_phi{i}", omega_phi[i])
        rama.addGlobalParameter(f"omega_psi{i}", omega_psi[i])
        rama.addGlobalParameter(f"phi0{i}", phi_i[i])
        rama.addGlobalParameter(f"psi0{i}", psi_i[i])
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type[i] == "IPR":
            rama.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]])
    rama.setForceGroup(forceGroup)
    return rama


def rama_proline_term(oa: OpenMMAWSEMSystem, 
                      k_rama_proline: float = 8.368, 
                      num_rama_proline_wells: int = 2, 
                      w: List[float] = [2.17, 2.15], 
                      sigma: List[float] = [105.52, 109.09], 
                      omega_phi: List[float] = [1.0, 1.0], 
                      phi_i: List[float] = [-1.153, -0.95], 
                      omega_psi: List[float] = [0.15, 0.15], 
                      psi_i: List[float] = [2.4, -0.218], 
                      forceGroup: int = 21) -> CustomCompoundBondForce:
    """Add Ramachandran potential for proline residues to the system.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k_rama_proline: The force constant for the Rama potential of proline. Default is 8.368.
        num_rama_proline_wells: The number of Ramachandran wells for proline. Default is 2.
        w: The weights for the Rama potential of proline. Default is [2.17, 2.15].
        sigma: The barrier heights for the Rama potential of proline. Default is [105.52, 109.09].
        omega_phi: The weights for the phi angle term of proline. Default is [1.0, 1.0].
        phi_i: The ideal phi angles for proline. Default is [-1.153, -0.95].
        omega_psi: The weights for the psi angle term of proline. Default is [0.15, 0.15].
        psi_i: The ideal psi angles for proline. Default is [2.4, -0.218].
        forceGroup: The force group to which this force will be added. Default is 21.

    Returns:
        A CustomCompoundBondForce object that implements the Rama potential for proline residues.
    """
    # add Rama potential for prolines
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama_proline *= oa.k_awsem
    rama_function = ''.join(["w_P%d*exp(-sigma_P%d*(omega_phi_P%d*phi_term%d^2+omega_psi_P%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i) for i in range(num_rama_proline_wells)])[:-1]
    rama_function = '-k_rama_proline*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0_P{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0_P{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_proline_wells)])
    rama_string = rama_function+rama_parameters
    rama = CustomCompoundBondForce(5, rama_string)

    if oa.periodic:
        rama.setUsesPeriodicBoundaryConditions(True)

    for i in range(num_rama_proline_wells):
        rama.addGlobalParameter(f"k_rama_proline", k_rama_proline)
        rama.addGlobalParameter(f"w_P{i}", w[i])
        rama.addGlobalParameter(f"sigma_P{i}", sigma[i])
        rama.addGlobalParameter(f"omega_phi_P{i}", omega_phi[i])
        rama.addGlobalParameter(f"omega_psi_P{i}", omega_psi[i])
        rama.addGlobalParameter(f"phi0_P{i}", phi_i[i])
        rama.addGlobalParameter(f"psi0_P{i}", psi_i[i])
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and oa.res_type[i] == "IPR":
            rama.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]])
    rama.setForceGroup(forceGroup)
    return rama


def rama_ssweight_term(oa: OpenMMAWSEMSystem, 
                       k_rama_ssweight: float = 8.368, 
                       num_rama_wells: int = 2, 
                       w: List[float] = [2.0, 2.0],
                       sigma: List[float] = [419.0, 15.398], 
                       omega_phi: List[float] = [1.0, 1.0], 
                       phi_i: List[float] = [-0.995, -2.25],
                       omega_psi: List[float] = [1.0, 1.0], 
                       psi_i: List[float] = [-0.82, 2.16], 
                       ssweight_file: str = "ssweight", 
                       forceGroup: int = 21):
    """Add Ramachandran SS potential with secondary structure weight to the system.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k_rama_ssweight: The force constant for the RamaSS term. Default is 8.368.
        num_rama_wells: The number of Ramachandran wells. Default is 2.
        w: The weights for the RamaSS potential. Default is [2.0, 2.0].
        sigma: The barrier heights for the RamaSS potential. Default is [419.0, 15.398].
        omega_phi: The weights for the phi angle term. Default is [1.0, 1.0].
        phi_i: The ideal phi angles for the RamaSS potential. Default is [-0.995, -2.25].
        omega_psi: The weights for the psi angle term. Default is [1.0, 1.0].
        psi_i: The ideal psi angles for the RamaSS potential. Default is [-0.82, 2.16].
        ssweight_file: The file containing secondary structure weights. Default is "ssweight".
        forceGroup: The force group to which this force will be added. Default is 21.

    Returns:
        A CustomCompoundBondForce object that implements the RamaSS potential with secondary structure weight.
    """
    # 8.368 = 2 * 4.184 kJ/mol, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_rama_ssweight *= oa.k_awsem
    rama_function = ''.join(["wSS%d*ssweight(%d,resId)*exp(-sigmaSS%d*(omega_phiSS%d*phi_term%d^2+omega_psiSS%d*psi_term%d^2))+" \
                            % (i, i, i, i, i, i, i) for i in range(num_rama_wells)])[:-1]
    rama_function = f'-{k_rama_ssweight}*(' + rama_function + ");"
    rama_parameters = ''.join([f"phi_term{i}=cos(phi_{i}-phi0SS{i})-1; phi_{i}=dihedral(p1, p2, p3, p4);\
                            psi_term{i}=cos(psi_{i}-psi0SS{i})-1; psi_{i}=dihedral(p2, p3, p4, p5);"\
                            for i in range(num_rama_wells)])
    rama_string = rama_function+rama_parameters
    ramaSS = CustomCompoundBondForce(5, rama_string)
    
    if oa.periodic:
        ramaSS.setUsesPeriodicBoundaryConditions(True)

    ramaSS.addPerBondParameter("resId")    
    for i in range(num_rama_wells):
        ramaSS.addGlobalParameter(f"wSS{i}", w[i])
        ramaSS.addGlobalParameter(f"sigmaSS{i}", sigma[i])
        ramaSS.addGlobalParameter(f"omega_phiSS{i}", omega_phi[i])
        ramaSS.addGlobalParameter(f"omega_psiSS{i}", omega_psi[i])
        ramaSS.addGlobalParameter(f"phi0SS{i}", phi_i[i])
        ramaSS.addGlobalParameter(f"psi0SS{i}", psi_i[i])
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
            ramaSS.addBond([oa.c[i-1], oa.n[i], oa.ca[i], oa.c[i], oa.n[i+1]], [i])
    ssweight = np.loadtxt(ssweight_file)
    ramaSS.addTabulatedFunction("ssweight", Discrete2DFunction(2, oa.nres, ssweight.flatten()))
    ramaSS.setForceGroup(forceGroup)
    return ramaSS


def side_chain_term(oa: OpenMMAWSEMSystem, 
                    k: Quantity=1*kilocalorie_per_mole, 
                    gmmFileFolder: str="/Users/weilu/opt/parameters/side_chain", 
                    forceGroup: int=25):
    """
    Add side chain chi angle forces to the system using Gaussian Mixture Models.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the side chain term. Default is 1 kcal/mol.
        gmmFileFolder: The directory path where the GMM parameters are stored. Default is "/Users/weilu/opt/parameters/side_chain".
        forceGroup: The force group to which this force will be added. Default is 25.

    Returns:
        A CustomCompoundBondForce object that implements the side chain chi angle forces.
    """
    # add chi forces
    # The sign of the equilibrium value is opposite and magnitude differs slightly
    # 251.04 = 60 * 4.184 kJ, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k = k.value_in_unit(kilojoule_per_mole)
    k_side_chain = k * oa.k_awsem
    n_components = 3

    means_all_res = np.zeros((20, 3, 3))
    precisions_chol_all_res = np.zeros((20, 3, 3, 3))
    log_det_all_res = np.zeros((20, 3))
    weights_all_res = np.zeros((20, 3))
    mean_dot_precisions_chol_all_res = np.zeros((20, 3, 3))

    res_type_map_letters = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                            'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    for i, res_type_one_letter in enumerate(res_type_map_letters):
        res_type = one_to_three[res_type_one_letter]
        if res_type == "GLY":
            weights_all_res[i] = np.array([1/3, 1/3, 1/3])
            continue

        means = np.loadtxt(f"{gmmFileFolder}/{res_type}_means.txt")
        precisions_chol = np.loadtxt(f"{gmmFileFolder}/{res_type}_precisions_chol.txt").reshape(3,3,3)
        log_det = np.loadtxt(f"{gmmFileFolder}/{res_type}_log_det.txt")
        weights = np.loadtxt(f"{gmmFileFolder}/{res_type}_weights.txt")
        means_all_res[i] = means

        precisions_chol_all_res[i] = precisions_chol
        log_det_all_res[i] = log_det
        weights_all_res[i] = weights


        for j in range(n_components):
            mean_dot_precisions_chol_all_res[i][j] = np.dot(means[j], precisions_chol[j])

    means_all_res = means_all_res.reshape(20, 9)
    precisions_chol_all_res = precisions_chol_all_res.reshape(20, 27)
    mean_dot_precisions_chol_all_res = mean_dot_precisions_chol_all_res.reshape(20, 9)

    log_weights = np.log(weights_all_res)
    sumexp_line = "+".join([f"exp(log_gaussian_and_weights_{i}-c)" for i in range(n_components)])
    const = 3 * np.log(2 * np.pi)
    side_chain = CustomCompoundBondForce(4, f"-{k_side_chain}*(log({sumexp_line})+c);\
                                        c=max(log_gaussian_and_weights_0,max(log_gaussian_and_weights_1,log_gaussian_and_weights_2));\
                                        log_gaussian_and_weights_0=log_gaussian_prob_0+log_weights(res,0);\
                                        log_gaussian_and_weights_1=log_gaussian_prob_1+log_weights(res,1);\
                                        log_gaussian_and_weights_2=log_gaussian_prob_2+log_weights(res,2);\
                                        log_gaussian_prob_0=-.5*({const}+log_prob_0)+log_det(res,0);\
                                        log_gaussian_prob_1=-.5*({const}+log_prob_1)+log_det(res,1);\
                                        log_gaussian_prob_2=-.5*({const}+log_prob_2)+log_det(res,2);\
                                        log_prob_0=((r1*pc(res,0)+r2*pc(res,3)+r3*pc(res,6)-mdpc(res,0))^2+\
                                        (r1*pc(res,1)+r2*pc(res,4)+r3*pc(res,7)-mdpc(res,1))^2+\
                                        (r1*pc(res,2)+r2*pc(res,5)+r3*pc(res,8)-mdpc(res,2))^2);\
                                        log_prob_1=((r1*pc(res,9)+r2*pc(res,12)+r3*pc(res,15)-mdpc(res,3))^2+\
                                        (r1*pc(res,10)+r2*pc(res,13)+r3*pc(res,16)-mdpc(res,4))^2+\
                                        (r1*pc(res,11)+r2*pc(res,14)+r3*pc(res,17)-mdpc(res,5))^2);\
                                        log_prob_2=((r1*pc(res,18)+r2*pc(res,21)+r3*pc(res,24)-mdpc(res,6))^2+\
                                        (r1*pc(res,19)+r2*pc(res,22)+r3*pc(res,25)-mdpc(res,7))^2+\
                                        (r1*pc(res,20)+r2*pc(res,23)+r3*pc(res,26)-mdpc(res,8))^2);\
                                        r1=10*distance(p1,p4);\
                                        r2=10*distance(p2,p4);\
                                        r3=10*distance(p3,p4)")
    
    if oa.periodic:
        side_chain.setUsesPeriodicBoundaryConditions(True)

    side_chain.addPerBondParameter("res")
    side_chain.addTabulatedFunction("pc", Discrete2DFunction(20, 27, precisions_chol_all_res.T.flatten()))
    side_chain.addTabulatedFunction("log_weights", Discrete2DFunction(20, 3, log_weights.T.flatten()))
    side_chain.addTabulatedFunction("log_det", Discrete2DFunction(20, 3, log_det_all_res.T.flatten()))
    side_chain.addTabulatedFunction("mdpc", Discrete2DFunction(20, 9, mean_dot_precisions_chol_all_res.T.flatten()))
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL":
            side_chain.addBond([oa.n[i], oa.ca[i], oa.c[i], oa.cb[i]], [gamma_se_map_1_letter[oa.seq[i]]])
    side_chain.setForceGroup(forceGroup)
    return side_chain


def chain_no_cb_constraint_term(oa: OpenMMAWSEMSystem, 
                                k_chain: float = 50208, 
                                bond_lengths: List[float] = [0.2459108, 0.2519591, 0.2466597], 
                                forceGroup: int = 20
                                ) -> HarmonicBondForce:
    """Apply a harmonic bond force to the system for residues without a C-beta atom.

    This function creates a harmonic bond force for the N-Ca and Ca-C bonds, as well as a Ca-Cb bond for the
    start and end residues of a chain, if they are not glycine or proline.

    Args:
        oa: An instance of OpenMMAWSEMSystem which provides access to the simulation system.
        k_chain: The force constant for the chain. Default is 50208 (kJ/mol/nm^2).
        bond_lengths: The equilibrium bond lengths for the N-Ca, Ca-C, and Ca-Cb bonds, respectively. Default is [0.2459108, 0.2519591, 0.2466597] nm.
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A HarmonicBondForce object that implements the chain constraints for residues without a C-beta atom.
    """
    # add chain forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_chain *= oa.k_awsem
    chain = HarmonicBondForce()

    if oa.periodic:
        chain.setUsesPeriodicBoundaryConditions(True)

    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends:
            chain.addBond(oa.n[i], oa.c[i], bond_lengths[2], k_chain)
    chain.setForceGroup(forceGroup)
    return chain


def con_no_cb_constraint_term(oa: OpenMMAWSEMSystem, 
                              k_con: float = 50208, 
                              bond_lengths: List[float] = [.3816, .240, .276, .153], 
                              forceGroup: int = 20
                              ) -> HarmonicBondForce:
    """
    Add constraints to the system for residues without a C-beta atom.

    This function creates a harmonic bond force for the N-Ca, Ca-O, and Ca-Ca+1 bonds, as well as a Ca-Cb bond for the
    start and end residues of a chain, if they are not glycine or proline.

    Args:
        oa: An instance of OpenMMAWSEMSystem which provides access to the simulation system.
        k_con: The force constant for the constraints. Default is 50208 (kJ/mol/nm^2).
        bond_lengths: The equilibrium bond lengths for the N-Ca, Ca-O, Ca-Ca+1, and Ca-Cb bonds, respectively. Default is [0.3816, 0.24, 0.276, 0.153] nm.
        forceGroup: The force group to which this force will be added. Default is 20.

    Returns:
        A HarmonicBondForce object that implements the constraints for residues without a C-beta atom.
    """
    # add con forces
    # 50208 = 60 * 2 * 4.184 * 100. kJ/nm^2, converted from default value in LAMMPS AWSEM
    # multiply interaction strength by overall scaling
    k_con *= oa.k_awsem
    con = HarmonicBondForce()

    if oa.periodic:
        con.setUsesPeriodicBoundaryConditions(True)
    
    for i in range(oa.nres):
        con.addBond(oa.ca[i], oa.o[i], bond_lengths[1], k_con)
        if ((i in oa.chain_starts) or (i in oa.chain_ends)) and (not oa.res_type[i] == "IGL"):
            # start doesn't have N, end doesn't have C. so only give a naive bond
            con.addBond(oa.ca[i], oa.cb[i], bond_lengths[3], k_con)
        if i not in oa.chain_ends:
            con.addBond(oa.ca[i], oa.ca[i+1], bond_lengths[0], k_con)
            con.addBond(oa.o[i], oa.ca[i+1], bond_lengths[2], k_con)
    con.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return con


def cbd_excl_term(oa: OpenMMAWSEMSystem, 
                  k: Quantity = 1*kilocalorie_per_mole, 
                  r_excl: float = 0.7, 
                  fileLocation: str = 'cbd_cbd_real_contact_symmetric.csv', 
                  forceGroup: int = 24
                  ) -> CustomNonbondedForce:
    """Apply a Cb domain excluded volume term to the system.

    This function creates a nonbonded force that penalizes steric clashes between Cb atoms
    of the protein, based on a residue-specific exclusion distance read from a CSV file.

    Args:
        oa: An instance of OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the excluded volume interaction. Default is 1 kcal/mol.
        r_excl: The cutoff distance for the excluded volume interaction in nanometers. Default is 0.7 nm.
        fileLocation: The file path to the CSV file containing the residue-specific exclusion distances. 
                      Default is 'cbd_cbd_real_contact_symmetric.csv'.
        forceGroup: The force group to which this force will be added. Default is 24.

    Returns:
        A CustomNonbondedForce object that implements the Cb domain excluded volume term.
    """
    # Cb domain Excluded volume
    # With residue specific parameters
    # a harmonic well with minimum at the database histogram peak.
    # and 1 kT(0.593 kcal/mol) penalty when the distance is at r_min of the database.
    # multiply interaction strength by overall scaling
    # Openawsem doesn't have the distance range (r_excl) change from 0.35 to 0.45 when the sequence separtation more than 5
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_excl = k * oa.k_awsem
    excl = CustomNonbondedForce(f"{k_excl}*step(r_max(res1,res2)-r)*((r-r_max(res1,res2))/(r_max(res1,res2)-r_min(res1,res2)))^2")
    excl.addPerParticleParameter("res")

    gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    for i in range(oa.natoms):
        excl.addParticle([gamma_se_map_1_letter[oa.seq[oa.resi[i]]]])
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)

    r_min_table = np.zeros((20,20))
    r_max_table = np.zeros((20,20))
    # fileLocation = '/Users/weilu/Research/server/mar_2020/cmd_cmd_exclude_volume/cbd_cbd_real_contact_symmetric.csv'
    df = pd.read_csv(fileLocation)
    for i, line in df.iterrows():
        res1 = line["ResName1"]
        res2 = line["ResName2"]
        r_min_table[gamma_se_map_1_letter[three_to_one[res1]]][gamma_se_map_1_letter[three_to_one[res2]]] = line["r_min"] / 10.0   # A to nm
        r_min_table[gamma_se_map_1_letter[three_to_one[res2]]][gamma_se_map_1_letter[three_to_one[res1]]] = line["r_min"] / 10.0
        r_max_table[gamma_se_map_1_letter[three_to_one[res1]]][gamma_se_map_1_letter[three_to_one[res2]]] = line["r_max"] / 10.0
        r_max_table[gamma_se_map_1_letter[three_to_one[res2]]][gamma_se_map_1_letter[three_to_one[res1]]] = line["r_max"] / 10.0

    excl.addTabulatedFunction("r_min", Discrete2DFunction(20, 20, r_min_table.T.flatten()))
    excl.addTabulatedFunction("r_max", Discrete2DFunction(20, 20, r_max_table.T.flatten()))
    excl.addInteractionGroup([x for x in oa.cb if x > 0], [x for x in oa.cb if x > 0])

    excl.setCutoffDistance(r_excl)
    if oa.periodic:
        excl.setNonbondedMethod(excl.CutoffPeriodic)
    else:
        excl.setNonbondedMethod(excl.CutoffNonPeriodic)

    # excl.setNonbondedMethod(excl.CutoffNonPeriodic)
    excl.createExclusionsFromBonds(oa.bonds, 1)
    excl.setForceGroup(forceGroup)
    print("cb domain exlcude volume term On")
    return excl
