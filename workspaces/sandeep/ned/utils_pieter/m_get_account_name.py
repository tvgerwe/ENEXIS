# Import system package.
import subprocess

# Define function.
def f_get_account_name() -> tuple:

    """
    Get name of the account.

    Parameters
    ----------
    -

    Returns
    -------
    str
        Account name.
    str
        Virtual environment name.
    """

    ###########################################################################
    # MAIN
    ###########################################################################
    
    try:
        # Get hardware UUID.
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Iterate through result to obtain hardware UUID. 
        for line in result.stdout.split("\n"):
            if "Hardware UUID" in line:
                c_uuid = line.split(":")[1].strip()
            
        # Determine account name, virtual environment name, en uuid.
        if c_uuid == '...':
            return 'macstudio', 've_macstudio'
        
        elif c_uuid == '92A825E9-D5DD-506B-9A46-7693F7C2DA65':
            return 'home', 've_macbook'

        else:
            raise ValueError(
                f"Unknown hardware UUID ({c_uuid}), cannot "
                f"determine accountname."
            )
        
    except subprocess.CalledProcessError as e:
        raise SystemError(f"Error retrieving hardware UUID: {e}")




