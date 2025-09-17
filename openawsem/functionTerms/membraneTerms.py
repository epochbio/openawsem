try:
    from openmm import CustomCompoundBondForce, CustomExternalForce
    from openmm.unit import nanometer, kilojoule_per_mole, angstrom, kilocalorie_per_mole, kilocalorie_per_mole, Quantity
except ModuleNotFoundError:
    from simtk.openmm import CustomCompoundBondForce, CustomExternalForce
    from simtk.unit import nanometer, kilojoule_per_mole, angstrom, kilocalorie_per_mole, kilocalorie_per_mole, Quantity
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
from openawsem.openAWSEM import OpenMMAWSEMSystem


def membrane_term(oa: OpenMMAWSEMSystem, 
                  k: Quantity = 1*kilocalorie_per_mole, 
                  k_m: float = 20, 
                  z_m: float = 1.5, 
                  membrane_center: Quantity = 0*angstrom, 
                  forceGroup: int = 24) -> CustomExternalForce:
    """Apply a membrane term to the system.

    This function creates a membrane force term that penalizes residues for being
    out of the membrane region, based on their hydrophobicity.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the membrane interaction. Default is 1 kcal/mol.
        k_m: The inverse decay length of the membrane interaction. Default is 20 nm^-1.
        z_m: Half of the membrane thickness in nanometers. Default is 1.5 nm.
        membrane_center: The z-coordinate of the membrane center plane. Default is 0 angstroms.
        forceGroup: The force group to which this force will be added. Default is 24.

    Returns:
        A CustomExternalForce object that implements the membrane term.
    """
    # k_m in units of nm^-1, z_m in units of nm.
    # z_m is half of membrane thickness
    # membrane_center is the membrane center plane shifted in z axis.
    # add membrane forces
    # 1 Kcal = 4.184 kJ strength by overall scaling
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_membrane = k * oa.k_awsem

    membrane = CustomExternalForce(f"k_membrane*\
            (0.5*tanh({k_m}*((z-{membrane_center})+{z_m}))+0.5*tanh({k_m}*({z_m}-(z-{membrane_center}))))*hydrophobicityScale")
    membrane.addPerParticleParameter("hydrophobicityScale")
    membrane.addGlobalParameter("k_membrane", k_membrane)
    zim = np.loadtxt("zim")
    # cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    ca = oa.ca
    for i in ca:
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
        membrane.addParticle(i, [zim[oa.resi[i]]])
    membrane.setForceGroup(forceGroup)
    return membrane


def membrane_with_pore_term(oa: OpenMMAWSEMSystem, 
                            k: Quantity=1*kilocalorie_per_mole, 
                            pore_center_x: float=0, 
                            pore_center_y: float=0, 
                            pore_radius: float=10, 
                            k_pore: float=0.1, 
                            k_m: float=20, 
                            z_m: float=1.5, 
                            membrane_center: Quantity=0*angstrom, 
                            forceGroup: int=24) -> CustomExternalForce:
    """Apply a membrane with a pore term to the system.

    This function creates a membrane force term that includes a pore, where the energy inside the pore is zero,
    as if the residue is in water. The pore is defined by its center coordinates and radius.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the membrane interaction. Default is 1 kcal/mol.
        pore_center_x: The x-coordinate of the pore center in nanometers. Default is 0 nm.
        pore_center_y: The y-coordinate of the pore center in nanometers. Default is 0 nm.
        pore_radius: The radius of the pore in nanometers. Default is 10 nm.
        k_pore: The force constant for the pore interaction. Default is 0.1.
        k_m: The inverse decay length of the membrane interaction. Default is 20 nm^-1.
        z_m: Half of the membrane thickness in nanometers. Default is 1.5 nm.
        membrane_center: The z-coordinate of the membrane center plane. Default is 0 angstroms.
        forceGroup: The force group to which this force will be added. Default is 24.

    Returns:
        A CustomExternalForce object representing the membrane with a pore interaction.
    """
    # inside the pore, the energy is zero, as if the residue is in the water.
    # pore_center_x, pore_center_y, pore_radius in unit of nanometer.

    # k_m in units of nm^-1, z_m in units of nm.
    # z_m is half of membrane thickness
    # membrane_center is the membrane center plane shifted in z axis.
    # add membrane forces
    # 1 Kcal = 4.184 kJ strength by overall scaling
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_membrane = k * oa.k_awsem

    membrane = CustomExternalForce(f"{k_membrane}*\
            (0.5*tanh({k_m}*((z-{membrane_center})+{z_m}))+0.5*tanh({k_m}*({z_m}-(z-{membrane_center}))))*(1-alpha)*hydrophobicityScale;\
            alpha=0.5*(1+tanh({k_pore}*({pore_radius}-rho)));\
            rho=((x-{pore_center_x})^2+(y-{pore_center_y})^2)^0.5")

    membrane.addPerParticleParameter("hydrophobicityScale")
    zim = np.loadtxt("zim")
    # cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    ca = oa.ca
    for i in ca:
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
        membrane.addParticle(i, [zim[oa.resi[i]]])
    membrane.setForceGroup(forceGroup)
    return membrane


def membrane_preassigned_term(oa: OpenMMAWSEMSystem, 
                              k: Quantity=1*kilocalorie_per_mole, 
                              k_m: float=20, 
                              z_m: float=1.5, 
                              membrane_center: Quantity=0*angstrom, 
                              zimFile: str="zimPosition", 
                              forceGroup: int=24
                              ) -> CustomExternalForce:
    """Apply a preassigned membrane term to the system.

    This function creates a membrane force term that depends on the preassigned zim positions
    of the residues, which are read from a file.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the membrane interaction. Default is 1 kcal/mol.
        k_m: The inverse decay length of the membrane interaction. Default is 20 nm^-1.
        z_m: Half of the membrane thickness. Default is 1.5 nm.
        membrane_center: The z-coordinate of the membrane center plane. Default is 0 angstroms.
        zimFile: The file name from which to read the zim positions. Default is "zimPosition".
        forceGroup: The force group to which this force will be added. Default is 24.

    Returns:
        A CustomExternalForce object representing the membrane interaction.
    """
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_membrane = k * oa.k_awsem

    membrane = CustomExternalForce(f"{k_membrane}*\
            (0.5*tanh({k_m}*((z-{membrane_center})+{z_m}))+0.5*tanh({k_m}*({z_m}-(z-{membrane_center}))))*zim")
    membrane.addPerParticleParameter("zim")
    # zim = np.loadtxt("zim")
    zimPosition = np.loadtxt(zimFile)
    zim = [-1 if z == 2 else 1 for z in zimPosition]
    # print(zim)
    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    for i in cb_fixed:
        membrane.addParticle(i, [zim[oa.resi[i]]])
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
    membrane.setForceGroup(forceGroup)
    return membrane


def SideToZ_m(side: str) -> float:
    """Convert the side of the membrane to a z-coordinate value.

    Args:
        side: A string indicating the side of the membrane ('up', 'down', or 'middle').

    Returns:
        A float representing the z-coordinate corresponding to the membrane side.
    """
    side = side.strip()
    if side == "down":
        return -1.5
    if side == "up":
        return 1.5
    if side == "middle":
        return 0


def membrane_preassigned_side_term(oa: OpenMMAWSEMSystem, 
                                  k: Quantity=1*kilocalorie_per_mole, 
                                  membrane_center: Quantity=0*angstrom, 
                                  zimFile: str="PredictedZimSide", 
                                  forceGroup: int=24
                                  ) -> CustomExternalForce:
    """Apply a preassigned side-dependent membrane term to the system.

    This function creates a membrane force term that depends on the side of the membrane
    each residue is preassigned to. The force is applied to the residues based on their
    zim file positions.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k: The force constant for the membrane interaction. Default is 1 kcal/mol.
        membrane_center: The z-coordinate of the membrane center. Default is 0 angstroms.
        zimFile: The filename of the zim positions file. Default is "PredictedZimSide".
        forceGroup: The force group to which this force will be added. Default is 24.

    Returns:
        A CustomExternalForce object that implements the preassigned side-dependent membrane term.
    """
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_membrane = k * oa.k_awsem
    membrane = CustomExternalForce(f"{k_membrane}*(abs(z-{membrane_center}-z_m))")
    membrane.addPerParticleParameter("z_m")

    with open(zimFile) as f:
        a = f.readlines()

    cb_fixed = [x if x > 0 else y for x,y in zip(oa.cb,oa.ca)]
    # print(cb_fixed)
    for i in cb_fixed:
        z_m = SideToZ_m(a[oa.resi[i]])
        # print(oa.resi[i])
        membrane.addParticle(i, [z_m])
    membrane.setForceGroup(forceGroup)
    return membrane


def single_helix_orientation_bias_term(oa: OpenMMAWSEMSystem, 
                                       k: Quantity=1*kilocalorie_per_mole, 
                                       membrane_center: Quantity=0*angstrom, 
                                       z_m: float=1.5, 
                                       k_m: float=20, 
                                       atomGroup: Union[int, List[int]]=-1, 
                                       forceGroup: int=18
                                       ) -> CustomCompoundBondForce:
    """
    Applies a bias to enforce the orientation of a single helix relative to the membrane.

    This function creates a force term that biases the orientation of a helix by penalizing deviations
    from a target z-position (membrane center) within a certain range defined by z_m.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        k: The force constant for the orientation bias. Default is 1 kcal/mol.
        membrane_center: The z-coordinate of the membrane center. Default is 0 angstroms.
        z_m: The half-thickness of the membrane, which defines the range of the orientation bias. Default is 1.5 nm.
        k_m: The steepness of the orientation bias potential. Default is 20.
        atomGroup: The group of atoms to which the orientation bias is applied. Default is -1, which implies all atoms.
        forceGroup: The force group to which this force will be added. Default is 18.

    Returns:
        A CustomCompoundBondForce object that implements the single helix orientation bias.
    """
    membrane_center = membrane_center.value_in_unit(nanometer)   # convert to nm
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_single_helix_orientation_bias = oa.k_awsem * k
    nres, ca = oa.nres, oa.ca
    if atomGroup == -1:
        group = list(range(nres))
    else:
        group = atomGroup
    n = len(group)
    theta_z1 = f"(0.5*tanh({k_m}*((z1-{membrane_center})+{z_m}))+0.5*tanh({k_m}*({z_m}-(z1-{membrane_center}))))"
    theta_z2 = f"(0.5*tanh({k_m}*((z2-{membrane_center})+{z_m}))+0.5*tanh({k_m}*({z_m}-(z2-{membrane_center}))))"
    normalization = n * (n - 1) / 2
    v_orientation = CustomCompoundBondForce(2, f"helix_orientation*{k_single_helix_orientation_bias}/{normalization}*((x1-x2)^2+(y1-y2)^2)*{theta_z1}*{theta_z2}")
    v_orientation.addGlobalParameter("helix_orientation", 1)
    # rcm_square = CustomCompoundBondForce(2, "1/normalization*(x1*x2)")
    # v_orientation.addGlobalParameter("k_single_helix_orientation_bias", k_single_helix_orientation_bias)
    # rg_square = CustomBondForce("1/normalization*(sqrt(x^2+y^2)-rcm))^2")
    # rg = CustomBondForce("1")
    # v_orientation.addGlobalParameter("normalization", n*n)
    for i in group:
        for j in group:
            if j <= i:
                continue
            v_orientation.addBond([ca[i], ca[j]], [])

    v_orientation.setForceGroup(forceGroup)
    return v_orientation
