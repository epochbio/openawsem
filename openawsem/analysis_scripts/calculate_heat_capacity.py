#!~/anaconda3/bin/python

import os
import pickle
import argparse
from cg_openmm.thermo.calc import get_heat_capacity
from openmm import unit


def main():
    """ This example demonstrates how to calculate heat capacity as a function of temperature from
    replica exchange energies, with uncertainties estimated using pyMBAR.

    Note: process_replica_exchange_data should first be run to determine the determine the start
    of the production region and energy decorrelation time."""

    parser = argparse.ArgumentParser(description="REMD Data Analysis: Metrics vs Temperature")
    parser.add_argument("-i", "--input_dir",
                        default = "../01_replica_exchange/output",
                        help = "path to folder where trajectories are stored."
                        "default is '../01_replica_exchange/output")
    parser.add_argument("-t","--traj",
                        default="output.nc",
                        help = "name of the trajectory output file. Default is" \
                        "output.nc")
    parser.add_argument("-s", "--analysis_stats",
                        default = "../01_replica_exchange/analysis_stats_discard_20ns.pkl",
                        help = "Path to pickled analysis of the trajecory" \
                        "created by 'process_remd.py'. Defaults to " \
                        "'../01_replica_exchange/analysis_stats_discard_20ns.pkl'")
    parser.add_argument("-o", "--output",
                        help = "Name to save heat capacity file to in format" \
                        " X_heat_capacity.png. If not set defaults to" \
                        "'heat_capacity.png")
    
    args = parser.parse_args()
    output_directory = args.input_dir
    output_data = os.path.join(output_directory, args.traj)

    if args.output:
        plot_name = "_".join([args.output, 'heat_capacity.png'])
    else:
        plot_name = 'heat_capacity.png'

    # Load in trajectory stats:
    analysis_stats = pickle.load(open(args.analysis_stats,"rb"))

    # Read the simulation coordinates for individual temperature replicas                                                                     
    C_v, dC_v, new_temperature_list, FWHM, Tm, Cv_height, N_eff = get_heat_capacity(
        output_data=output_data,
        frame_begin=analysis_stats["production_start"],
        sample_spacing=analysis_stats["energy_decorrelation"],
        num_intermediate_states=3,
        plot_file=plot_name,
    )

    print(f"T({new_temperature_list[0].unit})  Cv({C_v[0].unit})  dCv({dC_v[0].unit})")
    for i, C in enumerate(C_v):
        print(f"{new_temperature_list[i]._value:>8.2f}{C_v[i]._value:>10.4f} {dC_v[i]._value:>10.4f}")

if __name__ == "__main__":
    main()
