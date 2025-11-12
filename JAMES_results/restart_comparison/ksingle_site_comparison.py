dir_ai_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC_finalspin/run/220542.251106-161509_5year_AIrestart_run_with_h1_output'
dir_model_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC_finalspin/run/220243.251105-145758_5year_continous_run_1200_with_h1_output'

variable_list = ['RETRANSN', 'GPP', 'FPSN', 'AR', 'MR', 'GR', 'NPP', 'TLAI', 'LEAFC', 'DEADSTEMC', 'DEADCROOTC', 'FROOTC', 'LIVESTEMC', 'LIVECROOTC', 'TOTVEGC', 'N_ALLOMETRY','P_ALLOMETRY','BTRAN','QVEGE', 'QVEGT', 'FPG', 'FPG_P', 'CPOOL','NPOOL', 'PPOOL','CWDC', 'CWDN'] 
import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
	import xarray as xr
except ImportError:
	xr = None


def discover_h1_files(directory: str) -> List[str]:
	"""Return sorted list of NetCDF files containing elm.h1 in their names."""
	# Prefer a specific pattern to avoid other products
	pattern_specific = os.path.join(directory, '*elm.h1.*.nc')
	pattern_generic = os.path.join(directory, '*h1*.nc')
	files = sorted(glob.glob(pattern_specific))
	if not files:
		files = sorted(glob.glob(pattern_generic))
	return sort_files_by_timestamp_token(files)


def sort_files_by_timestamp_token(files: List[str]) -> List[str]:
	"""
	Sort files lexicographically by the token following '.h1.' which typically encodes a date like 0401-01-01-00000.
	Falls back to simple lexicographic filename sorting if token not found.
	"""
	def extract_token(p: str) -> Tuple[str, str]:
		name = os.path.basename(p)
		m = re.search(r'\.h1\.([^.]+)\.nc$', name)
		if m:
			return (m.group(1), name)
		return (name, name)
	return [p for _, _name, p in sorted([(extract_token(f)[0], extract_token(f)[1], f) for f in files])]


def reduce_to_time_series(data_array) -> pd.Series:
	"""
	Reduce a DataArray to a 1D time series by averaging across all non-time dimensions.
	Returns a pandas Series with a simple RangeIndex (we will reindex to day numbers later).
	"""
	if 'time' not in data_array.dims:
		# Try common alternative names, else raise
		for alt in ['Time', 't']:
			if alt in data_array.dims:
				data_array = data_array.rename({alt: 'time'})
				break
	if 'time' not in data_array.dims:
		raise ValueError('Variable has no time dimension.')
	other_dims = [d for d in data_array.dims if d != 'time']
	if other_dims:
		data_array = data_array.mean(dim=other_dims, skipna=True, keep_attrs=False)
	# Ensure 1D along time
	data_array = data_array.squeeze()
	if data_array.ndim != 1 or list(data_array.dims) != ['time']:
		raise ValueError('Unable to reduce variable cleanly to 1D time series.')
	values = data_array.values
	return pd.Series(values)


def extract_variable_timeseries(nc_files: List[str], var_name: str) -> Optional[pd.Series]:
	"""
	Open each file, extract the variable, reduce to 1D over time, and concatenate into a single series.
	Assumes each file has 365 timesteps, concatenates in file order.
	Returns None if the variable is missing from all files.
	"""
	if xr is None:
		raise RuntimeError('xarray is required but not installed.')
	series_parts: List[pd.Series] = []
	found_any = False
	for nc_path in nc_files:
		try:
			ds = xr.open_dataset(nc_path, decode_times=False)  # model calendars may be non-standard
		except Exception as e:
			print(f'[WARN] Failed to open {nc_path}: {e}')
			continue
		try:
			if var_name not in ds.variables:
				print(f'[INFO] Variable {var_name} not found in {os.path.basename(nc_path)}; skipping this file.')
				continue
			found_any = True
			da = ds[var_name]
			series_parts.append(reduce_to_time_series(da))
		except Exception as e:
			print(f'[WARN] Failed extracting {var_name} from {os.path.basename(nc_path)}: {e}')
		finally:
			ds.close()
	if not found_any or not series_parts:
		return None
	full = pd.concat(series_parts, ignore_index=True)
	# Index as day number starting at 1
	full.index = np.arange(1, len(full) + 1, dtype=int)
	full.index.name = 'day'
	return full


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
	csv_ai = base_out / 'csv' / 'ai'
	csv_model = base_out / 'csv' / 'model'
	csv_paired = base_out / 'csv' / 'paired'
	plots_dir = base_out / 'plots'
	for d in [csv_ai, csv_model, csv_paired, plots_dir]:
		d.mkdir(parents=True, exist_ok=True)
	return {'csv_ai': csv_ai, 'csv_model': csv_model, 'csv_paired': csv_paired, 'plots': plots_dir}


def save_series(series: pd.Series, target_csv: Path, value_col_name: str) -> None:
	df = pd.DataFrame({'day': series.index.values, value_col_name: series.values})
	df.to_csv(target_csv, index=False)
	print(f'[OK] Saved {target_csv}')


def save_paired(ai_series: pd.Series, model_series: pd.Series, target_csv: Path) -> None:
	max_len = max(len(ai_series), len(model_series))
	day = np.arange(1, max_len + 1, dtype=int)
	ai_vals = pd.Series(ai_series).reindex(day, fill_value=np.nan).values
	model_vals = pd.Series(model_series).reindex(day, fill_value=np.nan).values
	df = pd.DataFrame({'day': day, 'ai': ai_vals, 'model': model_vals})
	df.to_csv(target_csv, index=False)
	print(f'[OK] Saved {target_csv}')


def plot_paired(var_name: str, ai_series: pd.Series, model_series: pd.Series, target_png: Path) -> None:
	plt.figure(figsize=(10, 4))
	plt.plot(ai_series.index.values, ai_series.values, label='AI', linewidth=1.5)
	plt.plot(model_series.index.values, model_series.values, label='Model', linewidth=1.5)
	plt.xlabel('Day')
	plt.ylabel(var_name)
	plt.title(f'{var_name}: AI vs Model')
	plt.legend()
	plt.tight_layout()
	plt.savefig(target_png, dpi=150)
	plt.close()
	print(f'[OK] Saved {target_png}')


def main():
	import argparse

	parser = argparse.ArgumentParser(description='Extract, save, and plot paired time series from ELM h1 outputs.')
	parser.add_argument('--no-plot', action='store_true', help='Disable plot generation')
	parser.add_argument('--no-separate', action='store_true', help='Disable saving separate AI and model CSVs')
	parser.add_argument('--no-combined', action='store_true', help='Disable saving paired AI/Model combined CSVs')
	parser.add_argument('--limit-years', type=int, default=None, help='Limit number of files (years) per folder (default: all matched)')
	args = parser.parse_args()

	script_dir = Path(__file__).resolve().parent
	out_base = script_dir / 'outputs'
	paths = ensure_dirs(out_base)

	ai_files = discover_h1_files(dir_ai_result)
	model_files = discover_h1_files(dir_model_result)

	if args.limit_years:
		ai_files = ai_files[:args.limit_years]
		model_files = model_files[:args.limit_years]

	print(f'[INFO] Found {len(ai_files)} AI files and {len(model_files)} Model files.')
	if not ai_files or not model_files:
		print('[ERROR] Missing files in one or both directories; aborting.')
		return

	missing_vars = []
	for var in variable_list:
		print(f'[INFO] Processing variable: {var}')
		ai_series = extract_variable_timeseries(ai_files, var)
		model_series = extract_variable_timeseries(model_files, var)
		if ai_series is None and model_series is None:
			print(f'[WARN] Variable {var} missing in both sets; skipping.')
			missing_vars.append(var)
			continue
		if ai_series is None:
			print(f'[WARN] Variable {var} missing in AI set; plotting/saving model only where applicable.')
			ai_series = pd.Series(index=np.arange(1, len(model_series) + 1, dtype=int), dtype=float)
			ai_series.index.name = 'day'
		if model_series is None:
			print(f'[WARN] Variable {var} missing in Model set; plotting/saving AI only where applicable.')
			model_series = pd.Series(index=np.arange(1, len(ai_series) + 1, dtype=int), dtype=float)
			model_series.index.name = 'day'

		if not args.no_separate:
			save_series(ai_series, paths['csv_ai'] / f'{var}.csv', value_col_name='ai')
			save_series(model_series, paths['csv_model'] / f'{var}.csv', value_col_name='model')
		if not args.no_combined:
			save_paired(ai_series, model_series, paths['csv_paired'] / f'{var}.csv')
		if not args.no_plot:
			plot_paired(var, ai_series, model_series, paths['plots'] / f'{var}.png')

	if missing_vars:
		print(f'[INFO] Variables missing in both sets: {", ".join(missing_vars)}')


if __name__ == '__main__':
	main()
