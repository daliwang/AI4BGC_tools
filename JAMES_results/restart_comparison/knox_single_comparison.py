#dir_ai_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_cases/Daymet_ERA5_TESSFA2/uELM_knox_I1850CNPRDCTCBC_finalspin/run/222031.251111-130543_1year_daily_AI_alloutput'
#dir_model_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC_finalspin/run/220243.251105-145758_5year_continous_run_1200_with_h1_output'
#dir_model_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_cases/Daymet_ERA5_TESSFA2/uELM_knox_I1850CNPRDCTCBC_finalspin/run/222031.251111-130543_1year_daily_Modelrestart_run_with_h1_output'

#dir_model_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC_finalspin/run/221293.251107-230319_5year_Modelrestart_hourly_with_h1_output'
dir_model_result = '/gpfs/wolf2/cades/cli185/proj-shared/wangd/kmELM/e3sm_runs/uELM_knox_I1850CNPRDCTCBC_finalspin/run/222019.251111-124258_1year_daily_model_alloutput/'
dir_ai_result = './'

# the output direcory should be the output directory under the current directory

# Variable groups by stream:
# - First group (h1)
variable_list_h1 = ['RETRANSN', 'GPP', 'FPSN', 'AR', 'MR', 'GR', 'NPP', 'TLAI','LEAFC', 'DEADSTEMC', 'DEADCROOTC', 'FROOTC', 'LIVESTEMC', 'LIVECROOTC', 'TOTVEGC', 'N_ALLOMETRY','P_ALLOMETRY','BTRAN','QVEGE', 'QVEGT', 'FPG', 'FPG_P', 'CPOOL','NPOOL', 'PPOOL', 'SCALARAVG_vr']
# - Second group (h2)
variable_list_h2 = ['DEADCROOTC', 'DEADCROOTC_STORAGE', 'DEADCROOTN', 'DEADCROOTN_STORAGE', 'DEADCROOTP', 'DEADCROOTP_STORAGE', 'DEADSTEMC', 'DEADSTEMC_STORAGE', 'DEADSTEMN', 'DEADSTEMN_STORAGE', 'DEADSTEMP', 'DEADSTEMP_STORAGE', 'LEAFC', 'LEAFC_STORAGE', 'LEAFN', 'LEAFN_STORAGE', 'LEAFP', 'LEAFP_STORAGE', 'FROOTC', 'FROOTC_STORAGE', 'FROOTN', 'FROOTN_STORAGE', 'FROOTP', 'FROOTP_STORAGE', 'LIVESTEMC', 'LIVESTEMC_STORAGE', 'LIVECROOTC', 'LIVECROOTC_STORAGE', 'LIVECROOTN', 'LIVECROOTN_STORAGE', 'LIVECROOTP', 'LIVECROOTP_STORAGE', 'CPOOL', 'NPOOL', 'PPOOL', 'TLAI', 'TOTVEGC']
# - Third group (h3)
variable_list_h3 = ['CWDC_vr', 'CWDN_vr', 'CWDP_vr', 'LITR2C_vr', 'LITR3C_vr', 'LITR2N_vr', 'LITR3N_vr', 'LITR2P_vr', 'LITR3P_vr', 'SOIL1C_vr', 'SOIL1N_vr', 'SOIL1P_vr', 'SOIL2C_vr', 'SOIL2N_vr', 'SOIL2P_vr', 'SOIL3C_vr', 'SOIL3N_vr', 'SOIL3P_vr', 'SOIL4C_vr', 'SOIL4N_vr', 'SOIL4P_vr', 'LABILEP_vr', 'OCCLP_vr', 'PRIMP_vr', 'SECONDP_vr']
# Union list for monthly (h0) processing
variable_list = variable_list_h1 + variable_list_h2 + variable_list_h3

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
	return sort_files_by_timestamp_token(files, stream='h1')


def discover_h2_files(directory: str) -> List[str]:
	"""Return sorted list of NetCDF files containing elm.h2 in their names."""
	pattern_specific = os.path.join(directory, '*elm.h2.*.nc')
	pattern_generic = os.path.join(directory, '*h2*.nc')
	files = sorted(glob.glob(pattern_specific))
	if not files:
		files = sorted(glob.glob(pattern_generic))
	return sort_files_by_timestamp_token(files, stream='h2')


def discover_h3_files(directory: str) -> List[str]:
	"""Return sorted list of NetCDF files containing elm.h3 in their names."""
	pattern_specific = os.path.join(directory, '*elm.h3.*.nc')
	pattern_generic = os.path.join(directory, '*h3*.nc')
	files = sorted(glob.glob(pattern_specific))
	if not files:
		files = sorted(glob.glob(pattern_generic))
	return sort_files_by_timestamp_token(files, stream='h3')


def discover_h0_files(directory: str) -> List[str]:
	"""Return sorted list of NetCDF files containing elm.h0 (monthly) in their names."""
	pattern_specific = os.path.join(directory, '*elm.h0.*.nc')
	pattern_generic = os.path.join(directory, '*h0*.nc')
	files = sorted(glob.glob(pattern_specific))
	if not files:
		files = sorted(glob.glob(pattern_generic))
	return sort_files_by_timestamp_token(files, stream='h0')


def sort_files_by_timestamp_token(files: List[str], stream: str = 'h1') -> List[str]:
	"""
	Sort files lexicographically by the token following '.<stream>.' which typically encodes a date like 0401-01-01-00000.
	Falls back to simple lexicographic filename sorting if token not found.
	"""
	def extract_token(p: str) -> Tuple[str, str]:
		name = os.path.basename(p)
		try:
			pat = rf'\.{re.escape(stream)}\.([^.]+)\.nc$'
		except Exception:
			pat = r'\.h1\.([^.]+)\.nc$'
		m = re.search(pat, name)
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
	# Average across all non-time dimensions
	non_time_dims = [d for d in data_array.dims if d != 'time']
	if non_time_dims:
		data_array = data_array.mean(dim=non_time_dims, skipna=True, keep_attrs=False)
	# Drop any leftover size-1 dims other than time
	drop_dims = {d: 0 for d in data_array.dims if d != 'time' and data_array.sizes.get(d, 2) == 1}
	if drop_dims:
		data_array = data_array.isel(drop_dims)
	# Squeeze only non-time singleton dims; keep 'time' even if length-1 (monthly h0 case)
	dims_to_squeeze = [d for d in data_array.dims if d != 'time' and data_array.sizes.get(d, 2) == 1]
	if dims_to_squeeze:
		data_array = data_array.squeeze(dim=dims_to_squeeze)
	# Final validation
	if data_array.ndim != 1 or 'time' not in data_array.dims:
		raise ValueError(f'Unable to reduce variable cleanly to 1D time series. Dims: {data_array.dims}')
	values = data_array.values
	return pd.Series(values)


def _detect_time_resolution_seconds(ds) -> Optional[float]:
	"""
	Best-effort detection of time step in seconds from an xarray Dataset.
	Returns approximate step in seconds, or None if unknown.
	"""
	try:
		if 'time' not in ds.dims:
			return None
		time_var = ds['time']
		values = np.asarray(time_var.values).astype(float)
		if values.size < 2:
			return None
		step_raw = float(np.median(np.diff(values)))
		units = str(time_var.attrs.get('units', '')).lower()
		# Map CF-like time units to seconds
		sec_per_unit = 1.0
		if 'day' in units:
			sec_per_unit = 86400.0
		elif 'hour' in units:
			sec_per_unit = 3600.0
		elif 'minute' in units or 'min' in units:
			sec_per_unit = 60.0
		elif 'month' in units or 'mon' in units:
			# Approximate month length; sufficient for resolution labeling
			sec_per_unit = 30.0 * 86400.0
		elif 'second' in units or 'sec' in units:
			sec_per_unit = 1.0
		return step_raw * sec_per_unit
	except Exception:
		return None


def _resolution_label(step_seconds: Optional[float]) -> str:
	if step_seconds is None:
		return 'unknown'
	if 1800.0 <= step_seconds <= 7200.0:
		return 'hourly'
	if 0.5 * 86400.0 <= step_seconds <= 2.0 * 86400.0:
		return 'daily'
	# Roughly monthly (20 to 40 days)
	if 20.0 * 86400.0 <= step_seconds <= 40.0 * 86400.0:
		return 'monthly'
	return 'unknown'


def _detect_collection_resolution_seconds(nc_files: List[str]) -> Optional[float]:
	"""
	Detect representative time step (seconds) across a collection of files.
	Uses the median of detected steps across up to the first few files.
	"""
	steps: List[float] = []
	for nc_path in nc_files[:5]:
		try:
			ds = xr.open_dataset(nc_path, decode_times=False)
		except Exception:
			continue
		try:
			step = _detect_time_resolution_seconds(ds)
			if step is not None:
				steps.append(step)
		finally:
			ds.close()
	if not steps:
		return None
	return float(np.median(steps))


def extract_variable_timeseries(nc_files: List[str], var_name: str) -> Optional[pd.Series]:
	"""
	Open each file, extract the variable, reduce to 1D over time, and concatenate into a single series.
	Keeps native resolution (hourly or daily). Does not aggregate or resample.
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
	# Index as step number starting at 1
	full.index = np.arange(1, len(full) + 1, dtype=int)
	full.index.name = 'step'
	return full


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
	csv_ai = base_out / 'csv' / 'ai'
	csv_model = base_out / 'csv' / 'model'
	csv_paired = base_out / 'csv' / 'paired'
	plots_dir = base_out / 'plots'
	for d in [csv_ai, csv_model, csv_paired, plots_dir]:
		d.mkdir(parents=True, exist_ok=True)
	return {'csv_ai': csv_ai, 'csv_model': csv_model, 'csv_paired': csv_paired, 'plots': plots_dir}


def ensure_dirs_variant(base_out: Path, variant: str) -> Dict[str, Path]:
	"""
	Create output directories under a variant subfolder, e.g., 'monthly' or 'hourly'.
	"""
	return ensure_dirs(base_out / variant)


def save_series(series: pd.Series, target_csv: Path, value_col_name: str) -> None:
	df = pd.DataFrame({'step': series.index.values, value_col_name: series.values})
	df.to_csv(target_csv, index=False)
	print(f'[OK] Saved {target_csv}')


def save_paired(ai_series: pd.Series, model_series: pd.Series, target_csv: Path) -> None:
	max_len = max(len(ai_series), len(model_series))
	step = np.arange(1, max_len + 1, dtype=int)
	ai_vals = pd.Series(ai_series).reindex(step, fill_value=np.nan).values
	model_vals = pd.Series(model_series).reindex(step, fill_value=np.nan).values
	df = pd.DataFrame({'step': step, 'ai': ai_vals, 'model': model_vals})
	df.to_csv(target_csv, index=False)
	print(f'[OK] Saved {target_csv}')


def plot_paired(var_name: str, ai_series: pd.Series, model_series: pd.Series, target_png: Path, x_label: str = 'Step') -> None:
	plt.figure(figsize=(10, 4))
	# Plot model first (orange), then AI (blue) so legend order and visual layering match request
	plt.plot(model_series.index.values, model_series.values, label='Model', linewidth=1.5, color='tab:orange')
	plt.plot(ai_series.index.values, ai_series.values, label='AI', linewidth=1.5, color='tab:blue')
	plt.xlabel(x_label)
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
	parser.add_argument('--output-dir', type=str, default=None, help='Output base directory (default: ./outputs under current working directory)')
	parser.add_argument('--include-monthly', action='store_true', help='Also process monthly h0 files in addition to h1')
	parser.add_argument('--monthly-only', action='store_true', help='Process only monthly h0 files and skip h1')
	args = parser.parse_args()

	# Outputs under current working directory by default (or user-specified)
	if args.output_dir:
		out_base = Path(args.output_dir)
	else:
		out_base = Path.cwd() / 'outputs'
	paths = ensure_dirs(out_base)

	# h1 (hourly/daily) processing unless monthly-only
	if not args.monthly_only:
		ai_files = discover_h1_files(dir_ai_result)
		model_files = discover_h1_files(dir_model_result)

		# Detect resolution of each collection
		ai_step_seconds = _detect_collection_resolution_seconds(ai_files)
		model_step_seconds = _detect_collection_resolution_seconds(model_files)
		ai_res = _resolution_label(ai_step_seconds)
		model_res = _resolution_label(model_step_seconds)

		if args.limit_years:
			ai_files = ai_files[:args.limit_years]
			model_files = model_files[:args.limit_years]

		print(f'[INFO] Found {len(ai_files)} AI files ({ai_res}) and {len(model_files)} Model files ({model_res}) for h1.')
		if not ai_files or not model_files:
			print('[ERROR] Missing h1 files in one or both directories; skipping h1 block.')
		else:
			missing_vars = []
			for var in variable_list_h1:
				print(f'[INFO] (h1) Processing variable: {var}')
				ai_series = extract_variable_timeseries(ai_files, var)
				model_series = extract_variable_timeseries(model_files, var)
				if ai_series is None and model_series is None:
					print(f'[WARN] (h1) Variable {var} missing in both sets; skipping.')
					missing_vars.append(var)
					continue
				if ai_series is None:
					print(f'[WARN] (h1) Variable {var} missing in AI set; plotting/saving model only where applicable.')
					ai_series = pd.Series(index=np.arange(1, len(model_series) + 1, dtype=int), dtype=float)
					ai_series.index.name = 'step'
				if model_series is None:
					print(f'[WARN] (h1) Variable {var} missing in Model set; plotting/saving AI only where applicable.')
					model_series = pd.Series(index=np.arange(1, len(ai_series) + 1, dtype=int), dtype=float)
					model_series.index.name = 'step'

				if not args.no_separate:
					save_series(ai_series, paths['csv_ai'] / f'{var}.csv', value_col_name='ai')
					save_series(model_series, paths['csv_model'] / f'{var}.csv', value_col_name='model')
				# Only pair/plot when both collections share the same resolution
				if ai_res != 'unknown' and ai_res == model_res:
					if not args.no_combined:
						save_paired(ai_series, model_series, paths['csv_paired'] / f'{var}.csv')
					if not args.no_plot:
						xlab = 'Hour' if ai_res == 'hourly' else ('Day' if ai_res == 'daily' else 'Step')
						plot_paired(var, ai_series, model_series, paths['plots'] / f'{var}.png', x_label=xlab)
				else:
					print(f'[WARN] (h1) Skipping paired CSV/plot for {var} due to differing resolutions (AI={ai_res}, Model={model_res}).')

		# h2 (daily) processing
		ai_files_h2 = discover_h2_files(dir_ai_result)
		model_files_h2 = discover_h2_files(dir_model_result)
		if args.limit_years:
			ai_files_h2 = ai_files_h2[:args.limit_years]
			model_files_h2 = model_files_h2[:args.limit_years]
		h2_res_ai = _resolution_label(_detect_collection_resolution_seconds(ai_files_h2))
		h2_res_model = _resolution_label(_detect_collection_resolution_seconds(model_files_h2))
		print(f'[INFO] Found {len(ai_files_h2)} AI files ({h2_res_ai}) and {len(model_files_h2)} Model files ({h2_res_model}) for h2.')
		if not ai_files_h2 or not model_files_h2:
			print('[ERROR] Missing h2 files in one or both directories; skipping h2 block.')
		else:
			paths_h2 = ensure_dirs_variant(out_base, 'h2')
			for var in variable_list_h2:
				print(f'[INFO] (h2) Processing variable: {var}')
				ai_series = extract_variable_timeseries(ai_files_h2, var)
				model_series = extract_variable_timeseries(model_files_h2, var)
				if ai_series is None and model_series is None:
					print(f'[WARN] (h2) Variable {var} missing in both sets; skipping.')
					continue
				if ai_series is None:
					print(f'[WARN] (h2) Variable {var} missing in AI set; plotting/saving model only where applicable.')
					ai_series = pd.Series(index=np.arange(1, len(model_series) + 1, dtype=int), dtype=float)
					ai_series.index.name = 'step'
				if model_series is None:
					print(f'[WARN] (h2) Variable {var} missing in Model set; plotting/saving AI only where applicable.')
					model_series = pd.Series(index=np.arange(1, len(ai_series) + 1, dtype=int), dtype=float)
					model_series.index.name = 'step'
				if not args.no_separate:
					save_series(ai_series, paths_h2['csv_ai'] / f'{var}.csv', value_col_name='ai')
					save_series(model_series, paths_h2['csv_model'] / f'{var}.csv', value_col_name='model')
				if h2_res_ai != 'unknown' and h2_res_ai == h2_res_model:
					if not args.no_combined:
						save_paired(ai_series, model_series, paths_h2['csv_paired'] / f'{var}.csv')
					if not args.no_plot:
						xlab2 = 'Hour' if h2_res_ai == 'hourly' else ('Day' if h2_res_ai == 'daily' else 'Step')
						plot_paired(var, ai_series, model_series, paths_h2['plots'] / f'{var}.png', x_label=xlab2)
				else:
					print(f'[WARN] (h2) Skipping paired CSV/plot for {var} due to differing resolutions (AI={h2_res_ai}, Model={h2_res_model}).')

		# h3 (daily) processing
		ai_files_h3 = discover_h3_files(dir_ai_result)
		model_files_h3 = discover_h3_files(dir_model_result)
		if args.limit_years:
			ai_files_h3 = ai_files_h3[:args.limit_years]
			model_files_h3 = model_files_h3[:args.limit_years]
		h3_res_ai = _resolution_label(_detect_collection_resolution_seconds(ai_files_h3))
		h3_res_model = _resolution_label(_detect_collection_resolution_seconds(model_files_h3))
		print(f'[INFO] Found {len(ai_files_h3)} AI files ({h3_res_ai}) and {len(model_files_h3)} Model files ({h3_res_model}) for h3.')
		if not ai_files_h3 or not model_files_h3:
			print('[ERROR] Missing h3 files in one or both directories; skipping h3 block.')
		else:
			paths_h3 = ensure_dirs_variant(out_base, 'h3')
			for var in variable_list_h3:
				print(f'[INFO] (h3) Processing variable: {var}')
				ai_series = extract_variable_timeseries(ai_files_h3, var)
				model_series = extract_variable_timeseries(model_files_h3, var)
				if ai_series is None and model_series is None:
					print(f'[WARN] (h3) Variable {var} missing in both sets; skipping.')
					continue
				if ai_series is None:
					print(f'[WARN] (h3) Variable {var} missing in AI set; plotting/saving model only where applicable.')
					ai_series = pd.Series(index=np.arange(1, len(model_series) + 1, dtype=int), dtype=float)
					ai_series.index.name = 'step'
				if model_series is None:
					print(f'[WARN] (h3) Variable {var} missing in Model set; plotting/saving AI only where applicable.')
					model_series = pd.Series(index=np.arange(1, len(ai_series) + 1, dtype=int), dtype=float)
					model_series.index.name = 'step'
				if not args.no_separate:
					save_series(ai_series, paths_h3['csv_ai'] / f'{var}.csv', value_col_name='ai')
					save_series(model_series, paths_h3['csv_model'] / f'{var}.csv', value_col_name='model')
				if h3_res_ai != 'unknown' and h3_res_ai == h3_res_model:
					if not args.no_combined:
						save_paired(ai_series, model_series, paths_h3['csv_paired'] / f'{var}.csv')
					if not args.no_plot:
						xlab3 = 'Hour' if h3_res_ai == 'hourly' else ('Day' if h3_res_ai == 'daily' else 'Step')
						plot_paired(var, ai_series, model_series, paths_h3['plots'] / f'{var}.png', x_label=xlab3)
				else:
					print(f'[WARN] (h3) Skipping paired CSV/plot for {var} due to differing resolutions (AI={h3_res_ai}, Model={h3_res_model}).')

	# Monthly h0 processing
	if args.include_monthly or args.monthly_only:
		paths_monthly = ensure_dirs_variant(out_base, 'monthly')
		ai_h0 = discover_h0_files(dir_ai_result)
		model_h0 = discover_h0_files(dir_model_result)

		if args.limit_years:
			ai_h0 = ai_h0[:args.limit_years]
			model_h0 = model_h0[:args.limit_years]

		ai0_step = _detect_collection_resolution_seconds(ai_h0)
		model0_step = _detect_collection_resolution_seconds(model_h0)
		ai0_res = _resolution_label(ai0_step)
		model0_res = _resolution_label(model0_step)
		# Fallback: if resolution can't be detected but we're in the h0 stream, treat as monthly
		if ai0_res == 'unknown' and ai_h0:
			ai0_res = 'monthly'
		if model0_res == 'unknown' and model_h0:
			model0_res = 'monthly'

		print(f'[INFO] Found {len(ai_h0)} AI files ({ai0_res}) and {len(model_h0)} Model files ({model0_res}) for h0 (monthly).')
		if not ai_h0 or not model_h0:
			print('[ERROR] Missing h0 files in one or both directories; skipping monthly block.')
		else:
			missing_vars0 = []
			for var in variable_list:
				print(f'[INFO] (h0) Processing variable: {var}')
				ai_series0 = extract_variable_timeseries(ai_h0, var)
				model_series0 = extract_variable_timeseries(model_h0, var)
				if ai_series0 is None and model_series0 is None:
					print(f'[WARN] (h0) Variable {var} missing in both sets; skipping.')
					missing_vars0.append(var)
					continue
				if ai_series0 is None:
					print(f'[WARN] (h0) Variable {var} missing in AI set; plotting/saving model only where applicable.')
					ai_series0 = pd.Series(index=np.arange(1, len(model_series0) + 1, dtype=int), dtype=float)
					ai_series0.index.name = 'step'
				if model_series0 is None:
					print(f'[WARN] (h0) Variable {var} missing in Model set; plotting/saving AI only where applicable.')
					model_series0 = pd.Series(index=np.arange(1, len(ai_series0) + 1, dtype=int), dtype=float)
					model_series0.index.name = 'step'

				if not args.no_separate:
					save_series(ai_series0, paths_monthly['csv_ai'] / f'{var}.csv', value_col_name='ai')
					save_series(model_series0, paths_monthly['csv_model'] / f'{var}.csv', value_col_name='model')
				# For h0, pair/plot as monthly by default once both sides exist
				if ai0_res == model0_res:
					if not args.no_combined:
						save_paired(ai_series0, model_series0, paths_monthly['csv_paired'] / f'{var}.csv')
					if not args.no_plot:
						xlab0 = 'Month' if ai0_res == 'monthly' else ('Day' if ai0_res == 'daily' else ('Hour' if ai0_res == 'hourly' else 'Step'))
						plot_paired(var, ai_series0, model_series0, paths_monthly['plots'] / f'{var}.png', x_label=xlab0)
				else:
					print(f'[WARN] (h0) Skipping paired CSV/plot for {var} due to differing resolutions (AI={ai0_res}, Model={model0_res}).')


if __name__ == '__main__':
	main()
