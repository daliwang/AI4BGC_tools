import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# -- Define input directories --
dir1 = '20250502_US-MOz_ICB1850CNRDCTCBC_ad_spinup_trendy/run'
dir2 = '20250502_US-MOz_ICB1850CNPRDCTCBC_trendy/run'

# -- Get sorted list of .nc files --
# -- Helper to sort h0 files by year extracted from filename --
def sorted_h0_files(directory):
    files = glob.glob(os.path.join(directory, '*.elm.h0.*-01-01-00000.nc'))
    return sorted(files, key=lambda f: int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)))

#Ignore first file
files1 = sorted_h0_files(dir1)[1:]
files2 = sorted_h0_files(dir2)[1:]

# -- Load datasets --
ds1 = xr.open_mfdataset(files1, combine='nested', concat_dim='time')
ds2 = xr.open_mfdataset(files2, combine='nested', concat_dim='time')

# -- Extract years from filenames --
years1 = [int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)) for f in files1]
years2 = [int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)) for f in files2]
years2_offset = [y + max(years1) for y in years2]  # offset second series

# -- Assign time coordinates --
ds1 = ds1.assign_coords(time=years1)
ds2 = ds2.assign_coords(time=years2_offset)

# -- Concatenate datasets --
ds_combined = xr.concat([ds1, ds2], dim='time')

# -- Extract and spatially average NEE --
nee = ds_combined['NEE']
nee_mean = abs(nee.mean(dim=('lndgrid')))*24*3600*365  # or 'gridcell' if applicable

# -- Plot on log scale (handle negatives by shifting if needed) --
nee_min = nee_mean.min().compute().item()
if nee_min <= 0:
    offset = abs(nee_min) + 1e-6
    nee_log = np.log10(nee_mean + offset)
    ylabel = "log10(NEE + offset)"
else:
    nee_log = np.log10(nee_mean)
    ylabel = "log10(NEE)"

transition_year = max(years1)

# -- Plotting --
plt.figure(figsize=(10, 5))
plt.plot(ds_combined['time'], nee_log, marker='o')
plt.title("Log-scaled NEE Time Series")
plt.xlabel("Year")
plt.ylabel(ylabel)
# Horizontal line at y=0
plt.axhline(0, color='red', linestyle='--', label='NEE threshold (1 gC/m2/yr)')

# Vertical line at transition between dirs
plt.axvline(transition_year, color='blue', linestyle=':', label=f'Spinup transition')

plt.grid(True)
plt.show()
