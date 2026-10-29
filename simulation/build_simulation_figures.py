#!/usr/bin/env python3
"""Regenerate six simulation displays from saved summaries; no fitting."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(HERE/'build_superpopulation_figures.py')], check=True)
subprocess.run([sys.executable, str(HERE/'build_finite_w_figures.py'),
                '--input', str(HERE/'source_data/finite_w_summary.csv'),
                '--output', str(HERE)], check=True)
