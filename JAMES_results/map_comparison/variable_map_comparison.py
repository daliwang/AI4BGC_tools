import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import argparse
import glob


variables = ['CWDC', 'DEADSTEMC', 'GPP', 'HR', 'NEE', 'SOIL1C', 'SOIL2C', 'SOIL3C', 'SOIL4C', 'TLAI', 'TOTCOLC', 'TOTSOMC']
#variables = ['GPP']

def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def get_display_unit_and_scaled(data, var, ds):
    # Try to get area, landfrac, landmask
    area = ds["area"].values if "area" in ds.variables else 1.0
    landfrac = ds["landfrac"].values if "landfrac" in ds.variables else 1.0
    landmask = ds["landmask"].values if "landmask" in ds.variables else None
    units = var.attrs.get("units", "")
    display_unit = units
    # Mask
    if landmask is not None:
        mask = (landmask == 1)
        data = np.where(mask, data, np.nan)
    # Area/landfrac scaling
    weighted = data * area * landfrac
    # Unit conversion
    if units.endswith("gC/m^2/s"):
        weighted = weighted * 365 * 24 * 3600 / 1e6  # gC/m^2/s to gC/m^2/year, then to tonC/km^2/year
        display_unit = "tonC/km^2/year"
    elif units.endswith("gC/m^2"):
        weighted = weighted / 1e6  # gC/m^2 to tonC/km^2
        display_unit = "tonC/km^2"
    return weighted, display_unit


def _percent_diff_categories(model_vals: np.ndarray, ai_vals: np.ndarray) -> np.ndarray:
    """
    Categorize percent differences between model and AI values into 6 bins with sign information.
    
    Args:
        model_vals: Reference model values
        ai_vals: AI/updated values
        
    Returns:
        Categorized array with values:
        -2: -25%+ (large negative difference)
        -1: -10% to -25% (moderate negative difference)
         0: -10% to +10% (no significant difference)
        +1: +10% to +25% (moderate positive difference)
        +2: +25%+ (large positive difference)
    """
    model = np.asarray(model_vals, dtype=float)
    ai = np.asarray(ai_vals, dtype=float)
    cat = np.full(model.shape, np.nan, dtype=float)
    finite = np.isfinite(model) & np.isfinite(ai)
    if not np.any(finite):
        return cat
    
    # Threshold: zero-out percentage where |model| <= 10% of mean(|model|)
    mean_abs_model = np.nanmean(np.abs(model[finite])) if np.any(finite) else 0.0
    threshold = 0.1 * mean_abs_model
    denom = np.abs(model[finite])
    
    # Compute percent (AI - model)/model in %
    pct = (ai[finite] - model[finite]) / np.where(denom > 0, denom, 1.0) * 100.0
    
    # Apply threshold rule
    pct[denom <= threshold] = 0.0
    
    # Categorize into 5 bins with sign information
    bins = np.zeros_like(pct, dtype=int)
    
    # -2: -25%+, -1: -10% to -25%, 0: -10% to +10%, +1: +10% to +25%, +2: +25%+
    bins[(pct < -25)] = -2
    bins[(pct >= -25) & (pct < -10)] = -1
    bins[(pct >= -10) & (pct <= 10)] = 0
    bins[(pct > 10) & (pct <= 25)] = 1
    bins[(pct > 25)] = 2
    
    cat[finite] = bins.astype(float)
    return cat


def plot_variable_comparison(var, data1, data2, lon, lat, label1, label2, save_path, unit1, unit2):
    diff = data1 - data2
    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.3)
    proj = ccrs.PlateCarree()
    vmin_1 = float(np.nanmin(data1))
    vmax_1 = float(np.nanmax(data1))
    vmin_2 = float(np.nanmin(data2))
    vmax_2 = float(np.nanmax(data2))
    vmin_orig = min(vmin_1, vmin_2)
    vmax_orig = max(vmax_1, vmax_2)
    diff_abs_max = float(np.nanmax(np.abs(diff)))
    diff_min = float(np.nanmin(diff))
    diff_max = float(np.nanmax(diff))
    
    # Calculate statistics for summary
    stats = {
        'variable': var,
        'label1': label1,
        'label2': label2,
        'data1_min': vmin_1,
        'data1_max': vmax_1,
        'data1_sum': float(np.nansum(data1)),
        'data2_min': vmin_2,
        'data2_max': vmax_2,
        'data2_sum': float(np.nansum(data2)),
        'diff_min': diff_min,
        'diff_max': diff_max,
        'diff_sum': float(np.nansum(diff)),
        'diff_abs_max': diff_abs_max
    }
    
    # Helper function to add consistent map features
    def add_map_features(ax, title):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_global()
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
    
    # Helper function to add consistent colorbar
    def add_colorbar(im, ax, label, shrink=0.7, pad=0.05):
        cbar = plt.colorbar(im, ax=ax, shrink=shrink, pad=pad)
        cbar.set_label(label, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        return cbar
    
    # Plot 1: First dataset
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    im1 = ax1.pcolormesh(lon, lat, data1, transform=proj, vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    add_map_features(ax1, f'{var} - {label1}')
    cbar1 = add_colorbar(im1, ax1, f'{unit1}')
    
    # Plot 2: Second dataset
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    im2 = ax2.pcolormesh(lon, lat, data2, transform=proj, vmin=vmin_orig, vmax=vmax_orig, cmap='viridis')
    add_map_features(ax2, f'{var} - {label2}')
    cbar2 = add_colorbar(im2, ax2, f'{unit2}')
    
    # Plot 3: Absolute difference (removed max/min from title)
    ax3 = fig.add_subplot(gs[2, 0], projection=proj)
    norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
    im3 = ax3.pcolormesh(lon, lat, diff, transform=proj, norm=norm, cmap='RdBu_r')
    add_map_features(ax3, f'{var} - Difference ({label1} - {label2})')
    cbar3 = add_colorbar(im3, ax3, f'Δ{var}')
    
    # Plot 4: Percent-difference categorical map with red/blue colors
    ax4 = fig.add_subplot(gs[3, 0], projection=proj)
    # Treat data2 as model, data1 as AI/updated
    cat = _percent_diff_categories(data2, data1)
    
    # Define colors for the categorical map with red/blue scheme
    colors = [
        "#08519c",  # -2: -25%+ (dark blue - large negative difference)
        "#6baed6",  # -1: -10% to -25% (light blue - moderate negative difference)
        "#ffffff",  #  0: -10% to +10% (white - no significant difference)
        "#fcbba1",  # +1: +10% to +25% (light red - moderate positive difference)
        "#cb181d",  # +2: +25%+ (dark red - large positive difference)
    ]
    cmap = ListedColormap(colors)
    boundaries = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm_cat = BoundaryNorm(boundaries, cmap.N)
    
    im4 = ax4.pcolormesh(lon, lat, cat, transform=proj, cmap=cmap, norm=norm_cat)
    add_map_features(ax4, f'{var} - Percent Difference Categories')
    
    # Custom colorbar for categorical plot
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.7, pad=0.05, ticks=[-2, -1, 0, 1, 2])
    cbar4.set_label('Difference Category', fontsize=12, fontweight='bold')
    cbar4.ax.tick_params(labelsize=10)
    cbar4.ax.set_yticklabels([
        "-25%+", "-10% to -25%", "-10% to +10%", "+10% to +25%", "+25%+"
    ])
    
    # Ensure consistent layout and spacing
    plt.suptitle(f'Variable {var} Comparison Analysis', fontsize=18, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compare variables between NetCDF files.')
    parser.add_argument('--latest', type=str, default=None, help='Path to the latest result NetCDF file (file 1).')
    parser.add_argument('--previous', type=str, default=None, help='Path to a previous result NetCDF file (file 2).')
    parser.add_argument('--model', type=str, default='model_780_h0_results.nc', help='Path to the model reference NetCDF file.')
    parser.add_argument('--search_pattern', type=str, default='./*.nc', help='Glob pattern to search for NetCDF files.')
    args = parser.parse_args()

    # Find latest file if not provided
    if args.latest is None:
        args.latest = get_latest_file(args.search_pattern)
        print(f"Auto-selected latest file: {args.latest}")
    if args.previous is None:
        nc_files = sorted(glob.glob(args.search_pattern))
        prev_candidates = [f for f in nc_files if f != args.latest]
        if not prev_candidates:
            raise FileNotFoundError("No previous file found.")
        args.previous = prev_candidates[-1]
        print(f"Auto-selected previous file: {args.previous}")
    model_file = args.model
    
    # Extract base names from file paths for folder names
    latest_basename = os.path.splitext(os.path.basename(args.latest))[0]
    previous_basename = os.path.splitext(os.path.basename(args.previous))[0]
    model_basename = os.path.splitext(os.path.basename(model_file))[0]
    
    # Output folders using file names
    latest_model_dir = f'{latest_basename}_vs_{model_basename}'
    latest_reference_dir = f'{latest_basename}_vs_{previous_basename}'
    
    os.makedirs(latest_model_dir, exist_ok=True)
    os.makedirs(latest_reference_dir, exist_ok=True)
    
    # Open datasets
    ds_latest = xr.open_dataset(args.latest)
    ds_prev = xr.open_dataset(args.previous)
    ds_model = xr.open_dataset(model_file)
    lon = ds_latest['lon']
    lat = ds_latest['lat']
    
    print(f"\nComparing variables: {variables}\n")
    
    # Store all statistics for summary file
    all_stats = []
    
    for var in variables:
        if var not in ds_latest.data_vars or var not in ds_model.data_vars:
            print(f"Variable {var} not found in latest/model, skipping...")
        else:
            v_latest = ds_latest[var]
            v_model = ds_model[var]
            data_latest = v_latest.isel(time=0).values if 'time' in v_latest.dims else v_latest.values
            data_model = v_model.isel(time=0).values if 'time' in v_model.dims else v_model.values
            data_latest_scaled, unit_latest = get_display_unit_and_scaled(data_latest, v_latest, ds_latest)
            data_model_scaled, unit_model = get_display_unit_and_scaled(data_model, v_model, ds_model)
            save_path = os.path.join(latest_model_dir, f'{var}_comparison.png')
            stats = plot_variable_comparison(var, data_latest_scaled, data_model_scaled, lon, lat, 'Latest', 'Model', save_path, unit_latest, unit_model)
            stats['comparison_type'] = 'Latest vs Model'
            stats['unit1'] = unit_latest
            stats['unit2'] = unit_model
            all_stats.append(stats)
            
        if var not in ds_latest.data_vars or var not in ds_prev.data_vars:
            print(f"Variable {var} not found in latest/previous, skipping...")
        else:
            v_latest = ds_latest[var]
            v_prev = ds_prev[var]
            data_latest = v_latest.isel(time=0).values if 'time' in v_latest.dims else v_latest.values
            data_prev = v_prev.isel(time=0).values if 'time' in v_prev.dims else v_prev.values
            data_latest_scaled, unit_latest = get_display_unit_and_scaled(data_latest, v_latest, ds_latest)
            data_prev_scaled, unit_prev = get_display_unit_and_scaled(data_prev, v_prev, ds_prev)
            save_path = os.path.join(latest_reference_dir, f'{var}_comparison.png')
            stats = plot_variable_comparison(var, data_latest_scaled, data_prev_scaled, lon, lat, 'Latest', 'Previous', save_path, unit_latest, unit_prev)
            stats['comparison_type'] = 'Latest vs Previous'
            stats['unit1'] = unit_latest
            stats['unit2'] = unit_prev
            all_stats.append(stats)
    
    # Save summary statistics to CSV file
    import pandas as pd
    summary_file = f'comparison_summary_{latest_basename}.csv'
    df = pd.DataFrame(all_stats)
    df.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    ds_latest.close()
    ds_prev.close()
    ds_model.close()
    print("\nAll variable comparison plots are complete!")
    print(f"Images saved in: {latest_model_dir}/ and {latest_reference_dir}/")
    print(f"Summary file: {summary_file}")

if __name__ == '__main__':
    main()