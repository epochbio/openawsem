import matplotlib.pyplot as plt


def plot_ramachandran(df, frame_number):
    """
    Generate a Ramachandran plot for a specified frame from the simulation data.
    
    Args:
    df (pd.DataFrame): DataFrame containing the simulation angles data in long format.
    frame_number (int): The frame number to plot the Ramachandran plot for.
    
    """
    # Filter the DataFrame for the given frame number
    frame_data = df[df['Frame'] == frame_number]
    
    # Separate phi and psi angles
    phi_data = frame_data[frame_data['Angle_Type'] == 'Phi']['Angle']
    psi_data = frame_data[frame_data['Angle_Type'] == 'Psi']['Angle']
    
    # Ensure both phi and psi have the same number of entries
    if len(phi_data) != len(psi_data):
        raise ValueError("Mismatch in number of phi and psi angles for the frame.")
    
    # Create a new figure
    plt.figure(figsize=(6, 6))
    
    # Plot the trajectory data
    plt.scatter(phi_data, psi_data, color='blue', s=15, alpha=0.5)
    
    # Set x and y limits to -180 to 180 in 45 degree increments
    plt.xticks(ticks=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks(ticks=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    
    plt.xlabel('Phi angles (degrees)')
    plt.ylabel('Psi angles (degrees)')
    plt.title(f'Ramachandran Plot for Frame {frame_number}')
    plt.grid(True)
    
    # Plotting preferred regions for secondary structures can be added here if needed
    ### TODO: Add code to plot preferred regions for secondary structures

    plt.show()
