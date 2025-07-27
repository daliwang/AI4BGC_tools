import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

variable_names = ["NEE", "GPP", "HR", "TLAI", "TOTCOLC", "SOIL1C", 
"SOIL2C", "SOIL3C", "SOIL4C", "TOTSOMC", "CWDC", "DEADSTEMC"]  # Edit as needed

all_files = [f for f in os.listdir(".") if f.startswith("20250408_trendytest_ICB1850CNPRDCTCBC.elm.h0.") and f.endswith(".nc")]

if not all_files:
    print("No NetCDF files found in the current directory.")
    exit()

results = []

for variable_name in variable_names:
    twenty_year_sums = {}
    display_unit = ""
    for file_name in sorted(all_files):
        parts = file_name.split(".elm.h0.")
        if len(parts) > 1:
            date_str = parts[1][:4]
            if date_str.isdigit():
                year = int(date_str)
                file_path = os.path.join(".", file_name)
                if os.path.exists(file_path):
                    with xr.open_dataset(file_path) as ds:
                        if variable_name in ds.variables:
                            var = ds[variable_name]
                            area = ds["area"].values
                            landfrac = ds["landfrac"].values
                            landmask = ds["landmask"].values
                            units = var.attrs.get("units", "")
                            mask = (landmask == 1)
                            data = var.values
                            if data.ndim == 3:
                                data = np.nanmean(data, axis=0)
                            data = np.where(mask, data, np.nan)
                            weighted = data * area * landfrac
                            total = np.nansum(weighted)
                            if units.endswith("/s"):
                                total = total * 365 * 24 * 3600 / 1e6
                                display_unit = "tonC/year"
                            elif units.endswith("/m^2"):
                                total = total / 1e6
                                display_unit = "tonC"
                            twenty_year_sums[year] = total
    if not twenty_year_sums:
        print(f"No data found for variable {variable_name}. Check the variable name or input files.")
        continue
    years = sorted(twenty_year_sums.keys())
    sums_list = [twenty_year_sums[year] for year in years]
    results.append((variable_name, years, sums_list, display_unit))
    # Optionally, still write to file
    output_file = f"twenty_year_sum_{variable_name.lower()}_data.txt"
    with open(output_file, "w") as f:
        f.write("Year\tSum\n")
        for year, sum_val in zip(years, sums_list):
            f.write(f"{year:04d}\t{sum_val}\n")
    print(f"Aggregated {variable_name} data written to {output_file}")

# Plot all variables in subplots
n = len(results)
fig, axs = plt.subplots(n, 1, figsize=(10, 4*n), sharex=True)
if n == 1:
    axs = [axs]
for ax, (variable_name, years, sums_list, display_unit) in zip(axs, results):
    ax.plot(years, sums_list, marker="o", linestyle="-", color="b")
    ax.set_title(f"{variable_name} (aggregated) [{display_unit}]")
    ax.set_ylabel(f"{variable_name} ({display_unit})")
    ax.grid(True)
axs[-1].set_xlabel("Year")
plt.tight_layout()
fig.supxlabel("Year")  # Add a common x-axis label
plt.subplots_adjust(bottom=0.08)  # Adjust bottom margin if needed
folder_name = os.path.basename(os.getcwd())
output_png = f"trendy_all_variables_{folder_name}.png"
plt.savefig(output_png)
plt.close()
print(f"All plots saved in {output_png}")