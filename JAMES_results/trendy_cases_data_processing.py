import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse

# List of variables to process
variable_names = [
    "NEE", "GPP", "HR", "TLAI", "TOTCOLC", "SOIL1C",
    "SOIL2C", "SOIL3C", "SOIL4C", "TOTSOMC", "CWDC", "DEADSTEMC",
    "LABILEP", "SECONDP", "OCCLP", "PRIMP"
]

# Argument parser for optional directory
parser = argparse.ArgumentParser(description="Process NetCDF data in result directories.")
parser.add_argument('--dir', type=str, default=None, help='Process only this directory (e.g., run1). If not set, process all run* directories.')
args = parser.parse_args()

# Output directory
output_root = "restart_comparison"
os.makedirs(output_root, exist_ok=True)

# Get all subfolders in the current directory that start with 'run', or just the specified one
if args.dir:
    if not os.path.isdir(args.dir):
        raise ValueError(f"Specified directory {args.dir} does not exist or is not a directory.")
    folders = [args.dir]
else:
    folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith('run')]

for folder in folders:
    folder_path = os.path.join('.', folder)
    out_folder = os.path.join(output_root, folder)
    os.makedirs(out_folder, exist_ok=True)
    # Choose file pattern based on folder name
    if 'model' in folder:
        file_prefix = "20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0."
    else:
        file_prefix = "20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0."
    # Find all NetCDF files matching the pattern in this folder
    all_files = [f for f in os.listdir(folder_path)
                 if f.startswith(file_prefix) and f.endswith(".nc")]
    if not all_files:
        print(f"No NetCDF files found in {folder_path}.")
        continue
    for variable_name in variable_names:
        twenty_year_sums = {}
        display_unit = ""
        area_val = None
        for file_name in sorted(all_files):
            parts = file_name.split(".elm.h0.")
            if len(parts) > 1:
                date_str = parts[1][:4]
                if date_str.isdigit():
                    year = int(date_str)
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.exists(file_path):
                        with xr.open_dataset(file_path) as ds:
                            if variable_name in ds.variables:
                                var = ds[variable_name]

                                area = ds["area"].values
                                # print(f"area: {area}, units: {ds['area'].attrs.get('units', '')}")

                                landfrac = ds["landfrac"].values
                                landmask = ds["landmask"].values
                                units = var.attrs.get("units", "")
                                # print(f"variable: {variable_name},  Units: {units}")

                                mask = (landmask == 1)
                                data = var.values
                                if data.ndim == 3:
                                    data = np.nanmean(data, axis=0)
                                data = np.where(mask, data, np.nan)
                                weighted = data * area * landfrac
                                total = np.nansum(weighted)
                                print(f"total: {total}, units: {units}")

                                if units.endswith("gC/m^2/s"):
                                    total = total * 365 * 24 * 3600
                                    display_unit = "tonC/km^2/year"
                                elif units.endswith("gC/m^2"):
                                    total = total / 1e6 * 1e6
                                    display_unit = "tonC/km^2"
                                else:
                                    display_unit = units
                                twenty_year_sums[year] = total
                                if area_val is None:
                                    area_val = np.nansum(area * landfrac * mask)
        if not twenty_year_sums:
            print(f"No data found for variable {variable_name} in {folder}.")
            continue
        years = sorted(twenty_year_sums.keys())
        sums_list = [twenty_year_sums[year] for year in years]
        # Write to file
        output_file = os.path.join(out_folder, f"new_twenty_year_sum_{variable_name.lower()}_data.txt")

        with open(output_file, "w") as f:
            f.write(f"# Variable: {variable_name}\n")
            f.write(f"# Varibale Unit: {display_unit}\n")
            f.write(f"# Area units: {ds['area'].attrs.get('units', '')}\n")
            f.write("Year\tSum\n")
            for year, sum_val in zip(years, sums_list):
                f.write(f"{year:04d}\t{sum_val}\n")
        print(f"Aggregated {variable_name} data written to {output_file}")

print("All processing complete.")