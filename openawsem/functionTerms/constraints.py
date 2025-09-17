try:
    from openmm import CustomBondForce, CustomCentroidBondForce, CustomCVForce
    from openmm.unit import nanometer, kilojoule_per_mole, angstrom, kilocalorie_per_mole, Quantity
except ModuleNotFoundError:
    from simtk.openmm import CustomBondForce, CustomCentroidBondForce, CustomCVForce
    from simtk.unit import nanometer, kilojoule_per_mole, angstrom, kilocalorie_per_mole, Quantity
import numpy as np
from simtk.unit import dalton
from typing import List, Tuple, Union, Optional, Dict, Any
from openawsem.openAWSEM import OpenMMAWSEMSystem


def constraint_by_distance(oa: OpenMMAWSEMSystem, 
                           res1: int, 
                           res2: int, 
                           d0: Quantity=0*angstrom, 
                           forceGroup: int=3, 
                           k: Quantity=1*kilocalorie_per_mole
                           ) -> CustomBondForce:
    """Apply a harmonic constraint between two residues based on their distance.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        res1: The index of the first residue (0-indexed).
        res2: The index of the second residue (0-indexed).
        d0: The equilibrium distance between the two residues. Default is 0 angstroms.
        forceGroup: The force group to which this force will be added. Default is 3.
        k: The force constant for the harmonic constraint. Default is 1 kcal/mol.

    Returns:
        A CustomBondForce object that implements the constraint.
    """
    # print(len(oa.ca))
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomBondForce(f"0.5*{k_constraint}*(r-{d0})^2")
    # res1, res2 is 0 index. res1 = 0 means the first residue.
    constraint.addBond(*[oa.ca[res1], oa.ca[res2]])         # you could also do constraint.addBond(oa.ca[res1], oa.ca[res2])
    constraint.setForceGroup(forceGroup)
    return constraint

def group_constraint_by_distance(oa: OpenMMAWSEMSystem, 
                                 d0: Quantity=0*angstrom, 
                                 group1: List[int]=[oa.ca[0], oa.ca[1]], 
                                 group2: List[int]=[oa.ca[2], oa.ca[3]], 
                                 forceGroup: int=3, 
                                 k: Quantity=1*kilocalorie_per_mole) -> CustomCentroidBondForce:
    """Apply a harmonic constraint between two groups of particles based on their centroid distance.

    This function creates a constraint that acts on the centroids of two groups of particles,
    such as the alpha carbons of certain residues. The constraint is harmonic with a specified
    equilibrium distance and force constant. Note that CustomCentroidBondForce only works with
    the CUDA platform, not OpenCL.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        d0: The equilibrium distance between the centroids of the two groups. Default is 0 angstroms.
        group1: The first group of particles, typically a list of particle indices. Default is the first two alpha carbons.
        group2: The second group of particles, typically a list of particle indices. Default is the third and fourth alpha carbons.
        forceGroup: The force group to which this force will be added. Default is 3.
        k: The force constant for the harmonic constraint. Default is 1 kcal/mol.

    Returns:
        A CustomCentroidBondForce object that implements the constraint.
    """
    # CustomCentroidBondForce only work with CUDA not OpenCL.
    # only CA, CB, O has mass. so the group have to include those.
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomCentroidBondForce(2, f"0.5*{k_constraint}*(distance(g1,g2)-{d0})^2")
    # example group set up group1=[oa.ca[7], oa.cb[7]] use the ca and cb of residue 8.
    constraint.addGroup(group1)    # group use particle index.
    constraint.addGroup(group2)
    constraint.addBond([0, 1])
    constraint.setForceGroup(forceGroup)
    return constraint

def group_constraint_by_position(oa: OpenMMAWSEMSystem, 
                                 k: Quantity=1*kilocalorie_per_mole, 
                                 x0: float=10.0, 
                                 y0: float=10.0, 
                                 z0: float=10.0, 
                                 appliedToResidues: Union[int, List[int]]=-1, 
                                 forceGroup: int=3) -> CustomCVForce:
    """Apply a position-based constraint to a group of residues.

    This function applies a harmonic constraint that tethers a group of residues to a specified position in space.

    Args:
        oa: An OpenMMAWSEMSystem which provides access to the simulation system.
        k: The force constant for the harmonic constraint. Default is 1 kcal/mol.
        x0: The x-coordinate of the position to which the residues are tethered. Default is 10 nm.
        y0: The y-coordinate of the position to which the residues are tethered. Default is 10 nm.
        z0: The z-coordinate of the position to which the residues are tethered. Default is 10 nm.
        appliedToResidues: The indices of the residues to which the constraint is applied. Can be a single index or a list of indices. If set to -1, the constraint is applied to all residues. Default is -1.
        forceGroup: The force group to which this force will be added. Default is 3.

    Returns:
        A CustomCVForce object that implements the position-based constraint.
    """
    # x0, y0, z0 is in unit of nm.
    # appliedToResidues can be a list of residue index. for example appliedToResidues=[0, 1], to tether the first two residues.
    # 1 Kcal = 4.184 kJ strength by overall scaling
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    sum_of_x_coord = CustomExternalForce(f"x*mass")
    sum_of_y_coord = CustomExternalForce(f"y*mass")
    sum_of_z_coord = CustomExternalForce(f"z*mass")

    sum_of_x_coord.addPerParticleParameter("mass")
    sum_of_y_coord.addPerParticleParameter("mass")
    sum_of_z_coord.addPerParticleParameter("mass")

    # print("index for CAs", oa.ca)
    print(f"mass can be retrieved as ", oa.system.getParticleMass(oa.ca[0]))
    total_mass = 0.0
    for i in range(oa.natoms):
        if appliedToResidues == -1:
            mass = oa.system.getParticleMass(i).value_in_unit(dalton)
            sum_of_x_coord.addParticle(i, [mass])
            sum_of_y_coord.addParticle(i, [mass])
            sum_of_z_coord.addParticle(i, [mass])
            total_mass += mass
        elif oa.resi[i] in appliedToResidues:
            mass = oa.system.getParticleMass(i).value_in_unit(dalton)
            sum_of_x_coord.addParticle(i, [mass])
            sum_of_y_coord.addParticle(i, [mass])
            sum_of_z_coord.addParticle(i, [mass])
            total_mass += mass
        # if oa.resi[i] == appliedToResidue:
        #     pulling.addParticle(i)
        # print(oa.resi[i] , oa.seq[oa.resi[i]])
    print(f"total_mass = {total_mass}")
    harmonic = CustomCVForce(f"{k_constraint}*((sum_x/{total_mass}-{x0})^2+(sum_y/{total_mass}-{y0})^2+(sum_z/{total_mass}-{z0})^2)")
    harmonic.addCollectiveVariable("sum_x", sum_of_x_coord)
    harmonic.addCollectiveVariable("sum_y", sum_of_y_coord)
    harmonic.addCollectiveVariable("sum_z", sum_of_z_coord)
    harmonic.setForceGroup(forceGroup)
    return harmonic
