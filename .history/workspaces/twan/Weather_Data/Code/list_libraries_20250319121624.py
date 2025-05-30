import os
import re

# Define the path to the repository
repo_path = '/Users/Twan/Library/Mobile Documents/com~apple~CloudDocs/Data Science/Data Projects EASI/ENEXIS-2'

# Set to store unique library names
libraries = set()

# Regular expressions to match import statements
import_re = re.compile(r'^\s*import (\S+)')
from_import_re = re.compile(r'^\s*from (\S+) import')

# Walk through the repository
for root, _, files in os.walk(repo_path):
    for file in files:
        if file.endswith('.py'):
            with open(os.path.join(root, file), 'r') as f:
                for line in f:
                    match = import_re.match(line) or from_import_re.match(line)
                    if match:
                        libraries.add(match.group(1))

# Print the list of libraries
for lib in sorted(libraries):
    print(lib)