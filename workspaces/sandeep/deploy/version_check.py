import subprocess

# List of dependencies to check for
dependencies = [
    "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "prefect",
    "griffe", "rich", "statsmodels", "xgboost", "joblib", "polars", "icecream",
    "tensorflow", "prophet", "pytorch", "torchvision", "torchaudio", "transformers"
]

def get_installed_packages():
    # Execute the 'conda list' command to get installed packages in the current environment
    try:
        result = subprocess.run(['conda', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        output = result.stdout.decode('utf-8')
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while fetching installed packages: {e}")
        return ""

def check_versions():
    # Get installed packages in the active conda environment
    installed_packages = get_installed_packages()
    
    print(f"Checking installed versions in the current environment:\n")
    
    # Check if the package is in the list and print the version
    for package in dependencies:
        if package in installed_packages:
            # Search for the package and extract the version from the conda list output
            start_idx = installed_packages.find(package)
            end_idx = installed_packages.find('\n', start_idx)
            package_info = installed_packages[start_idx:end_idx]
            
            # Split the package info and extract the version
            version = package_info.split()[1]
            print(f"{package} Version: {version}")
        else:
            print(f"{package} is not installed.")

if __name__ == "__main__":
    check_versions()
