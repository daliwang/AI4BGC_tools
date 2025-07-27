import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Get the variable name as input
variable_name = input("Enter the variable name (e.g., NEE): ")

# Initialize variables
annual_values = {}
years = []

# Get all NetCDF files in the current directory
all_files = [f for f in os.listdir(".") if f.endswith(".nc")]

# Filter files based on the year range (501 to 921)
filtered_files = []
for file_name in all_files:
    # Extract the year and month from the file name (assuming the format "YYYY-MM" after ".elm.h0.")
    parts = file_name.split(".elm.h0.")
    if len(parts) > 1:
        date_str = parts[1][:7]  # Extract the first 7 characters after ".elm.h0." (e.g., "YYYY-MM")
        if len(date_str) == 7 and date_str[:4].isdigit() and date_str[5:7].isdigit():
            year = int(date_str[:4])
            if 501 <= year <= 921:
                filtered_files.append(file_name)

# Process each file
for file_name in sorted(filtered_files):  # Sort files by year and month
    # Extract the year from the file name
    year = int(file_name.split(".elm.h0.")[1][:4])

    # Open the NetCDF file and read the specified variable
    file_path = os.path.join(".", file_name)
    if os.path.exists(file_path):
        with xr.open_dataset(file_path) as ds:
            if variable_name in ds.variables:
                # Sum the variable values over all dimensions (if applicable)
                monthly_value = np.sum(ds[variable_name].values)
                if year not in annual_values:
                    annual_values[year] = 0
                annual_values[year] += monthly_value
            else:
                print(f"{variable_name} variable not found in {file_name}")
    else:
        print(f"File not found: {file_name}")

# Prepare data for writing and plotting
years = sorted(annual_values.keys())
annual_values_list = [annual_values[year] for year in years]

# Write the annual values to a file
output_file = f"annual_{variable_name.lower()}_data_501_921.txt"
with open(output_file, "w") as f:
    f.write("Year\tAnnual Value\n")
    for year, value in zip(years, annual_values_list):
        f.write(f"{year:04d}\t{value}\n")

print(f"Annual {variable_name} data written to {output_file}")

# Plot the annual values
plt.figure(figsize=(10, 6))
plt.plot(years, annual_values_list, marker="o", linestyle="-", color="b")
plt.title(f"Annual {variable_name} Data (501-921)")
plt.xlabel("Year")
plt.ylabel(f"Annual {variable_name}")
plt.grid(True)
plt.show()