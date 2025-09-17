from openawsem.functionTerms import *
from openawsem.helperFunctions.myFunctions import *

try:
    from openmm import Force
    from openmm.unit import angstrom, Quantity
except ModuleNotFoundError:
    from simtk.openmm import Force
    from simtk.unit import angstrom, Quantity
from typing import List
from openawsem.openAWSEM import OpenMMAWSEMSystem


def set_up_forces(oa: OpenMMAWSEMSystem, 
                  computeQ: bool = False, 
                  submode: int = -1, 
                  contactParameterLocation: str = ".", 
                  membrane_center: Quantity = -0*angstrom
                  ) -> List[Force]:
    """Set up the forces for the OpenAWSEM simulation.

    Args:
        oa: An instance of OpenMMAWSEMSystem, which provides access to the simulation system.
        computeQ: A boolean indicating whether to compute the Q value. Defaults to False.
        submode: An integer representing the submode for additional forces. Defaults to -1.
        contactParameterLocation: A string representing the location of the contact parameter files. Defaults to ".".
        membrane_center: A Quantity specifying the center of the membrane. Defaults to -0*angstrom.

    Returns:
        A list of Force objects that have been set up for the simulation.
    """
    # apply forces
    forces = [
        ### Basic Forces ###
        basicTerms.con_term(oa),
        basicTerms.chain_term(oa),
        basicTerms.chi_term(oa),
        basicTerms.excl_term(oa, periodic=False),
        basicTerms.rama_term(oa),
        basicTerms.rama_proline_term(oa),
        # basicTerms.rama_ssweight_term(oa, k_rama_ssweight=2*8.368),

        ### Contact Forces ###
        contactTerms.contact_term(oa),  # <- optimize?
        # for membrane protein simulation use contact_term below.
        # contactTerms.contact_term(oa, z_dependent=True, inMembrane=True, membrane_center=membrane_center, k_relative_mem=3),

        ### Hydrogen Bond Forces ###
        hydrogenBondTerms.beta_term_1(oa),
        hydrogenBondTerms.beta_term_2(oa),
        hydrogenBondTerms.beta_term_3(oa),
        hydrogenBondTerms.pap_term_1(oa),
        hydrogenBondTerms.pap_term_2(oa),

        ### Membrane Forces ###
        # membraneTerms.membrane_term(oa, k=1*kilocalorie_per_mole, membrane_center=membrane_center),
        # membraneTerms.membrane_preassigned_term(oa, k=1*kilocalorie_per_mole, membrane_center=membrane_center, zimFile="PredictedZim"),

        ### Biasing Forces ###
        # templateTerms.er_term(oa),
        # templateTerms.fragment_memory_term(oa, frag_file_list_file="./frags.mem", npy_frag_table="./frags.npy", UseSavedFragTable=True),
        templateTerms.fragment_memory_term(oa, frag_file_list_file="./single_frags.mem", npy_frag_table="./single_frags.npy", UseSavedFragTable=False),

        ### Constraint Forces ###
        debyeHuckelTerms.debye_huckel_term(oa, chargeFile="charge.txt"),
    ]
    if computeQ:
        forces.append(biasTerms.rg_term(oa))
        forces.append(biasTerms.q_value(oa, "crystal_structure-cleaned.pdb", forceGroup=1))
        # forces.append(qc_value(oa, "crystal_structure-cleaned.pdb"))
        # forces.append(partial_q_value(oa, "crystal_structure-cleaned.pdb", residueIndexGroup=list(range(0, 15)), forceGroup=1))
    if submode == 0:
        additional_forces = [
            # contact_term(oa),
        ]
        forces += additional_forces
    return forces
