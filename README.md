# AI4BGC Tools

This repository contains a collection of Python scripts and utilities for processing, analyzing, and visualizing biogeochemical model outputs, particularly from NetCDF files. The tools are organized into two main directories: `others/` and `JAMES_results/`.

## Directory Structure

- `others/` — General-purpose scripts for NetCDF data analysis and visualization.
- `JAMES_results/` — Scripts and results related to the JAMES project, including data processing, comparison, and plotting utilities.

## Requirements

- Python 3.x
- numpy
- xarray
- matplotlib
- netCDF4
- cartopy (for map plotting)

Install dependencies with:
```bash
pip install numpy xarray matplotlib netCDF4 cartopy
```

## Scripts Overview

### others/
- **tva_plotspinup_zdr.py**: Concatenates and analyzes NEE (Net Ecosystem Exchange) from two simulation runs, applies log-scaling, and visualizes the time series with transition markers.
- **trendplot3.py**: Aggregates and plots multiple variables (e.g., NEE, GPP, HR) from NetCDF files, saving results and plots for each variable.
- **tvaplot.py**: Sums a user-specified variable over a year range (501–921) from NetCDF files and plots annual values.
- **Show2DVariables.v2.py**: Visualizes 2D variables from NetCDF files using a mask, allowing interactive variable selection and subdomain plotting.
- **ncdiff2.py**: Compares variables between two NetCDF files, reporting differences in data, shape, and type.

### JAMES_results/
- **trendy_cases_data_processing.py**: Processes NetCDF data in result directories, aggregates variables, and writes summary files for further analysis.
- **map_comparison/variable_map_comparison.py**: Compares spatial maps of variables between two datasets, visualizing differences using Cartopy.
- **restart_comparison/trendy_case_comparison_plot.py**: Plots and compares time series of variables across multiple simulation runs, saving combined plots for each variable.

## Usage

1. Place your NetCDF files in the appropriate directory as required by each script.
2. Run scripts from the command line. For example:
   ```bash
   python others/tva_plotspinup_zdr.py
   python others/trendplot3.py
   python JAMES_results/trendy_cases_data_processing.py --dir run1
   ```
3. Some scripts require user input (e.g., variable name or file paths).
4. Output plots and summary files will be saved in the current or specified output directory.

## Notes
- Edit directory paths and file patterns in scripts as needed to match your data organization.
- For plotting scripts, ensure you have a display or use a backend that supports file output (e.g., PNG).

## License
Specify your license here (e.g., MIT, GPL, etc.).

## Contact
For questions or contributions, please contact the repository maintainer. 