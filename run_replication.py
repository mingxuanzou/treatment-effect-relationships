#!/usr/bin/env python3
"""Reproduce simulations, applications, and manuscript displays from bundled inputs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['all','simulations','applications','figures','preflight'],nargs='?',default='all')
    args=parser.parse_args()
    if args.mode=='preflight':
        run('targeted_revision/super/code/run_all.py','--preflight')
        run('x_interactions_revision/code/run_comparison.py','--preflight')
        return
    if args.mode in ('all','simulations'):
        run('targeted_revision/super/code/run_all.py','--refresh-plot-data')
        run('x_interactions_revision/code/run_comparison.py')
        shutil.copyfile(ROOT/'x_interactions_revision/results/summary.csv',ROOT/'simulation/source_data/finite_w_summary.csv')
    if args.mode in ('all','applications'):
        run('prepare_data.py')
        run('real_x_comparison/nvss/code/run.py')
        run('real_x_comparison/hillstrom/code/run_comparison.py')
        run('real_x_comparison/audit/validate.py')
    run('build_display_set.py')
    run('real_x_comparison/write_manuscript_tables.py')
    print('Replication complete. See tables_figures.pdf and real_x_comparison/tables/*.tex.')
if __name__=='__main__':main()
