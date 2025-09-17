import mdtraj as md
import numpy as np
import pandas as pd


def extract_angles_from_traj(traj: md.Trajectory, top: str, outfile: str) -> None:
    """
    Extracts phi, psi, and omega angles from a given trajectory and saves the data to a CSV file.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory from which to extract angles. This is an MDTraj Trajectory object.
    top : str
        The topology file associated with the trajectory. This is not used in the function but included for API consistency.
    outfile : str
        The path to the output CSV file where the angle data will be saved.

    Returns
    -------
    angles_df: pd.DataFrame
        DataFrame with the bb rotational angles in degrees.
    """
    # Load your trajectory using MDTraj
    traj = md.load(traj, top=top)

    # Calculate phi and psi angles for each frame
    phi_angles = md.compute_phi(traj)
    psi_angles = md.compute_psi(traj)
    omega_angles = md.compute_omega(traj)

    # phi_angles and psi_angles are tuples containing (indices, values)
    # indices are the atom indices used to calculate each angle
    # values are the actual angle values for each frame in radians

    # You might want to convert angles from radians to degrees
    phi_degrees = np.degrees(phi_angles[1])  # Convert to degrees
    psi_degrees = np.degrees(psi_angles[1])
    omega_degrees = np.degrees(omega_angles[1])

    # Since the first residue does not have a phi angle, prepend a NaN to each frame's phi array
    phi_degrees = np.concatenate([np.full((phi_degrees.shape[0], 1), np.nan), phi_degrees], axis=1)

    # Since the last residue does not have a psi angle, append a NaN to each frame's psi array
    psi_degrees = np.concatenate([psi_degrees, np.full((psi_degrees.shape[0], 1), np.nan)], axis=1)

    # Create a DataFrame for each type of angle
    num_frames, num_residues = phi_degrees.shape
    frame_indices = np.repeat(np.arange(num_frames), num_residues)
    residue_indices = np.tile(np.arange(num_residues) + 1, num_frames)  # residues are typically 1-indexed in biochemistry


    phi_df = pd.DataFrame({
        'Frame': frame_indices,
        'Residue': residue_indices,
        'Angle_Type': 'Phi',
        'Angle': phi_degrees.flatten()
    })

    psi_df = pd.DataFrame({
        'Frame': frame_indices,
        'Residue': residue_indices,
        'Angle_Type': 'Psi',
        'Angle': psi_degrees.flatten()
    })

    omega_df = pd.DataFrame({
        'Frame': frame_indices,
        'Residue': residue_indices,
        'Angle_Type': 'Omega',
        'Angle': omega_degrees.flatten()
    })

    # Concatenate the dataframes to form a single long-format DataFrame
    angles_df = pd.concat([phi_df, psi_df, omega_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    angles_df.to_csv(outfile, index=False, float_format='%.3f', sep='\t')
    return(angles_df)


def load_angles(infile: str) -> pd.DataFrame:
    """
    Loads angle data from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    infile : str
        The path to the input CSV file containing angle data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing angle data.
    """
    return pd.read_csv(infile, sep='\t')
