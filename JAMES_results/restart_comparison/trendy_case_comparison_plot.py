# This script scans all folders in the restart_comparison directory, collects all unique variable names from file names,
# and for each variable, generates a combined line plot of the data across all folders.
# Each plot is saved as a PNG file named after the variable in the restart_comparison folder.
#
# The data files are expected to be named in the format: new_twenty_year_sum_{variable}_data.txt
# Each file contains data in the format of Year\tSum, with header lines specifying the variable name and unit.
# The x-axis is the Year, and the y-axis is the Sum of the variable (with the display unit from the file header).
# The script excludes data for year 0001 from the plots.

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# the path of the restart_comparison folder
restart_comparison_path = "/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/JAMES_results/restart_comparison"

# get all the folders in the restart_comparison_path
folders = [f for f in os.listdir(restart_comparison_path) if os.path.isdir(os.path.join(restart_comparison_path, f))]

# Step 1: Collect all unique variable names from files in all folders
variable_set = set()
file_pattern_re = re.compile(r"new_twenty_year_sum_(.+?)_data\.txt")

for folder in folders:
    folder_path = os.path.join(restart_comparison_path, folder)
    for fname in os.listdir(folder_path):
        match = file_pattern_re.match(fname)
        if match:
            variable_set.add(match.group(1))

# Step 2: For each variable, plot the combined line plot across all folders
for variable in variable_set:
    file_pattern = f"new_twenty_year_sum_{variable}_data.txt"
    plt.figure()
    plot_title = None
    y_label = None
    for folder in folders:
        file_path = os.path.join(restart_comparison_path, folder, file_pattern)
        if os.path.exists(file_path):
            # Read the data, skipping header lines
            with open(file_path) as f:
                lines = f.readlines()
            # Extract variable name and unit from the first two lines
            var_line = lines[0].strip()
            unit_line = lines[1].strip()
            variable_name = var_line.split(":", 1)[-1].strip() if ":" in var_line else variable.upper()
            unit = unit_line.split(":", 1)[-1].strip() if ":" in unit_line else ""
            if plot_title is None:
                plot_title = f"{variable_name} Comparison Across Folders"
            if y_label is None:
                y_label = unit
            data = np.loadtxt(file_path, skiprows=4)
            # Exclude year 0001
            filtered_data = data[data[:, 0] != 1]
            plt.plot(filtered_data[:, 0], filtered_data[:, 1], label=folder)
    plt.xlabel("Year")
    plt.ylabel(y_label if y_label else "Sum")
    plt.title(plot_title if plot_title else f"{variable.upper()} Comparison Across Folders")
    plt.legend()
    plt.savefig(os.path.join(restart_comparison_path, f"{variable.upper()}_comparison.png"))
    plt.close()     