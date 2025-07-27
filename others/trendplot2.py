import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# Get the variable name as input
variable_name = input("Enter the variable name (e.g., NEE): ")

# Initialize variables
twenty_year_sums = {}

# Get all NetCDF files in the current directory
all_files = [f for f in os.listdir(".") if f.startswith("20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.") and f.endswith(".nc")]

if not all_files:
    print("No NetCDF files found in the current directory.")
    exit()

# Process each file
for file_name in sorted(all_files):  # Sort files by year
    print(f"Processing file: {file_name}")
    # Extract the year from the file name (e.g., "0201" from "20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.0201-01-01-00000.nc")
    parts = file_name.split(".elm.h0.")
    if len(parts) > 1:
        date_str = parts[1][:4]  # Extract the first 4 characters (e.g., "0201")
        if date_str.isdigit():
            year = int(date_str)
            # Open the NetCDF file and read the specified variable
            file_path = os.path.join(".", file_name)
            if os.path.exists(file_path):
                with xr.open_dataset(file_path) as ds:
                    if variable_name in ds.variables:
                        print(f"Found variable: {variable_name} in {file_name}")
                        # Calculate the sum of the variable over all dimensions
                        twenty_year_sum = np.nansum(ds[variable_name].values)
                        twenty_year_sums[year] = twenty_year_sum
                    else:
                        print(f"Variable {variable_name} not found in {file_name}")
            else:
                print(f"File not found: {file_name}")

# Prepare data for writing
if not twenty_year_sums:
    print(f"No data found for variable {variable_name}. Check the variable name or input files.")
    exit()

years = sorted(twenty_year_sums.keys())
sums_list = [twenty_year_sums[year] for year in years]

# Write the 20-year aggregated values to a file
output_file = f"twenty_year_sum_{variable_name.lower()}_data.txt"
with open(output_file, "w") as f:
    f.write("Year\t20-Year Sum\n")
    for year, sum in zip(years, sums_list):
        f.write(f"{year:04d}\t{sum}\n")

print(f"20-Year aggregated {variable_name} data written to {output_file}")

# Plot the time-series values
plt.figure(figsize=(10, 6))
plt.plot(years, sums_list, marker="o", linestyle="-", color="b")
plt.title(f" Trendy {variable_name} Data")
plt.xlabel("Year")
plt.ylabel(f"Spatially aggregated {variable_name}")
plt.grid(True)
plt.show()