def print_last_two_sims(folder):
    """ To know which parameters were used in the simualtion you are plotting.
    
    Parameters
    ----------
    folder : str
        The folder where the simulation data is stored.
    """

    filename = folder+"commandline_args.txt"
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
            # Print the last two lines
            print(''.join(lines[-2:]))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")