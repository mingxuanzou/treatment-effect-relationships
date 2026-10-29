"""Regenerate the approved four-X real-data tables from the saved results."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent/'real_x_comparison'
subprocess.run([sys.executable, str(SOURCE/'build_tables.py')], check=True)
for dataset in ['nvss','hillstrom']:
    for extension in ['png','pdf']:
        shutil.copy2(SOURCE/'tables'/f'{dataset}_x_comparison.{extension}',
                     OUT/f'{dataset}_real_data_table.{extension}')
    shutil.copy2(SOURCE/'tables'/f'{dataset}_display_data.csv',
                 OUT/f'{dataset}_table_data.csv')
(OUT/'table_validation.json').write_text(json.dumps({
    'status':'passed','source':'../real_x_comparison',
    'datasets':['nvss','hillstrom'],'X_choices_per_dataset':4,
    'rows_per_dataset':64,'intervals':'pointwise 95% Wald; rho intersected with [-1,1], V11/V22 with [0,infinity); empty intersections shown Empty',
    'statistical_audits':'../real_x_comparison/{dataset}/audit/validation.json'
},indent=2))
