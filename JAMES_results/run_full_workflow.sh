#!/bin/bash
set -e  # Exit on any error

# --- CONFIGURE THESE PATHS ---
ARCHIVE_DIR="/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_spinup/e3sm_run/20250408_trendytest_ICB1850CNPRDCTCBC/run"
DATA_PROCESSING_DIR="/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/JAMES_results"
RESTART_COMPARISON_DIR="/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/JAMES_results/restart_comparison"

# --- STEP 1: Run archive_script.py ---
echo "Running archive_script.py..."
cd "$ARCHIVE_DIR"
python archive_script.py

# --- STEP 2: Run data_processing.py ---
echo "Running data_processing.py..."
cd "$DATA_PROCESSING_DIR"
python data_processing.py

# --- STEP 3: Run case_compare_plot.py ---
echo "Running case_compare_plot.py..."
cd "$RESTART_COMPARISON_DIR"
python case_compare_plot.py

echo "Workflow complete!"