#!/usr/bin/env python3
try:
    from openmm import LangevinIntegrator, Platform, CustomIntegrator
    from openmm.app import Simulation, PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter
    from openmm.unit import picosecond, picoseconds, femtoseconds, kelvin
except ModuleNotFoundError:
    from simtk.openmm import LangevinIntegrator, Platform, CustomIntegrator
    from simtk.openmm.app import Simulation, PDBReporter, DCDReporter, StateDataReporter, CheckpointReporter
    from simtk.unit import picosecond, picoseconds, femtoseconds, kelvin

import os
import sys
import time
from random import seed, randint
import argparse
import importlib.util

from openawsem import *
from openawsem.helperFunctions.myFunctions import *
from openmmtools.multistate import ReplicaExchangeSampler
from openmmtools.multistate import MultiStateReporter
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.mcmc import LangevinDynamicsMove

# Global variables for convenience
do = os.system
cd = os.chdir

def run_replica_exchange(args):
    """Run the replica exchange simulation with the given arguments."""

    simulation_platform = args.platform
    platform = Platform.getPlatformByName(simulation_platform)
    if simulation_platform == "CPU":
        if args.thread != -1:
            platform.setPropertyDefaultValue("Threads", str(args.thread))
        print(f"{simulation_platform}: {platform.getPropertyDefaultValue('Threads')} threads")

    setupFolderPath = os.path.dirname(args.protein)
    setupFolderPath = "." if setupFolderPath == "" else setupFolderPath
    proteinName = pdb_id = os.path.basename(args.protein)
    
    pwd = os.getcwd()
    toPath = os.path.abspath(args.to)
    checkPointPath = None if args.fromCheckPoint is None else os.path.abspath(args.fromCheckPoint)
    forceSetupFile = None if args.forces is None else os.path.abspath(args.forces)
    parametersLocation = "." if args.parameters is None else os.path.abspath(args.parameters)
    os.chdir(setupFolderPath)

    chain = args.chain
    pdb = f"{pdb_id}.pdb"
    if chain == "-1":
        chain = getAllChains("crystal_structure.pdb")
        print("Chains to simulate: ", chain)

    if args.to != "./":
        os.makedirs(toPath, exist_ok=True)
        os.system(f"cp {forceSetupFile} {toPath}/forces_setup.py")
        os.system(f"cp crystal_structure.fasta {toPath}/")
        os.system(f"cp crystal_structure.pdb {toPath}/")

    if args.fromOpenMMPDB:
        input_pdb_filename = proteinName
        seq=read_fasta("crystal_structure.fasta")
        print(f"Using Seq:\n{seq}")
    else:
        suffix = '-openmmawsem.pdb'
        if pdb_id[-len(suffix):] == suffix:
            input_pdb_filename = pdb_id
        else:
            input_pdb_filename = f"{pdb_id}-openmmawsem.pdb"
        seq=None

    if args.fasta == "":
        seq = None
    else:
        seq = read_fasta(args.fasta)
        print(f"Using Seq:\n{seq}")

    # --- REMD Specific Setup ---

    # Define temperature range for replicas
    num_replicas = args.replicas
    temps = np.linspace(args.tempStart, args.tempEnd, num_replicas) * kelvin

    # Initialize AWSEM system
    oa = OpenMMAWSEMSystem(input_pdb_filename, k_awsem=1.0, chains=chain, xml_filename=openawsem.xml, seqFromPdb=seq, includeLigands=args.includeLigands)
    spec = importlib.util.spec_from_file_location("forces", forceSetupFile)
    forces_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forces_module)
    myForces = forces_module.set_up_forces(oa, submode=args.subMode, contactParameterLocation=parametersLocation)
    oa.addForcesWithDefaultForceGroup(myForces)

    # Minimize the initial structure
    integrator = CustomIntegrator(0.001)
    simulation = Simulation(oa.pdb.topology, oa.system, integrator, platform)
    simulation.context.setPositions(oa.pdb.positions)
    simulation.minimizeEnergy()
    positions = simulation.context.getState(getPositions=True).getPositions()
    
    # Create the thermodynamic states for each replica
    thermodynamic_states = []
    for temp in temps:
        integrator = LangevinIntegrator(temp, 1/picosecond, args.timeStep*femtoseconds)
        # Note: We create a new integrator for each temperature.
        # OpenMMTools will handle the rest.
        state = ThermodynamicState(system=oa.system, temperature=temp)
        thermodynamic_states.append(state)

    # Create the sampler states (initial positions and velocities for each replica)
    sampler_states = []
    for _ in temps:
        state = SamplerState(positions=positions)
        sampler_states.append(state)

    # Set up the reporter to save data
    reporter = MultiStateReporter(os.path.join(toPath, "output.nc"), checkpoint_interval=args.reportFrequency)

    # # Create and run the replica exchange sampler
    # sampler = ReplicaExchangeSampler(
    #     mcmc_moves=LangevinIntegrator(1/picosecond, 1/picosecond, args.timeStep*femtoseconds),
    #     number_of_replicas=num_replicas,
    #     reporter=reporter,
    # )

     # Create and run the replica exchange sampler
    sampler = ReplicaExchangeSampler(
        mcmc_moves=LangevinDynamicsMove(
            timestep=args.timeStep*femtoseconds,
            collision_rate=1/picosecond,
        ),
        number_of_replicas=num_replicas,
        reporter=reporter,
    )
    sampler.temperature_trajectories = temps
    sampler.states = thermodynamic_states
    sampler.sampler_states = sampler_states
    sampler.platform = platform
    sampler.report_interval = args.reportFrequency
    sampler.n_steps = int(args.steps)
    sampler.run()

    # Log time taken
    end_time = time.time()
    time_taken = end_time - start_time
    hours, rest = divmod(time_taken, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"--- {hours} hours {minutes} minutes {seconds} seconds ---")
    
    timeFile = os.path.join(toPath, "time.dat")
    with open(timeFile, "w") as out:
        out.write(str(time_taken) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run OpenAWSEM Replica Exchange MD simulation.")
    parser.add_argument("protein", help="The name of the protein")
    parser.add_argument("--to", default="./", help="location of output files")
    parser.add_argument("-c", "--chain", type=str, default="-1")
    parser.add_argument("-t", "--thread", type=int, default=-1, help="default is using all available")
    parser.add_argument("-p", "--platform", type=str, default="OpenCL")
    parser.add_argument("-s", "--steps", type=float, default=2e4, help="number of steps per replica, default 1e5")
    parser.add_argument("--tempStart", type=float, default=280, help="Starting temperature")
    parser.add_argument("--tempEnd", type=float, default=562, help="Ending temperature")
    parser.add_argument("--replicas", type=int, default=12, help="Number of replicas")
    parser.add_argument("--fromCheckPoint", type=str, default=None, help="The checkpoint file you want to start from")
    parser.add_argument("--subMode", type=int, default=-1)
    parser.add_argument("-f", "--forces", default="forces_setup.py")
    parser.add_argument("--parameters", default=None)
    parser.add_argument("-r", "--reportFrequency", type=int, default=1000, help="Frequency to save data to output.nc")
    parser.add_argument("--fromOpenMMPDB", action="store_true", default=False)
    parser.add_argument("--fasta", type=str, default="crystal_structure.fasta")
    parser.add_argument("--timeStep", type=int, default=2)
    parser.add_argument("--includeLigands", action="store_true", default=False)
    args = parser.parse_args()

    with open('commandline_args.txt', 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    print(' '.join(sys.argv))

    run_replica_exchange(args)

if __name__ == "__main__":
    main()
