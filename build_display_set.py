#!/usr/bin/env python3
"""Regenerate the requested six figures and two real-data tables."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import subprocess
import sys
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
scripts=[] if '--merge-only' in sys.argv else ['simulation/build_simulation_figures.py']
if '--reuse-tables' not in sys.argv and '--merge-only' not in sys.argv:
    scripts.append('tables/build_real_tables.py')
for script in scripts:
    subprocess.run([sys.executable, str(ROOT/script)], check=True)
pages = [ROOT/'simulation'/f'{framework}_{metric}.pdf'
         for framework in ['superpopulation','finite_population']
         for metric in ['coverage','bias','rmse']]
pages += [ROOT/'tables'/f'{name}_real_data_table.pdf' for name in ['nvss','hillstrom']]
writer = PdfWriter()
for path in pages:
    reader = PdfReader(path)
    assert len(reader.pages)==1, path
    writer.append(reader)
buffer=BytesIO()
writer.write(buffer)
payload=buffer.getvalue()
assert payload.rstrip().endswith(b'%%EOF')
temporary=ROOT/'tables_figures.pdf.tmp'
temporary.write_bytes(payload)
temporary.replace(ROOT/'tables_figures.pdf')
assert len(PdfReader(ROOT/'tables_figures.pdf').pages)==8
manifest = {'pages': [{'page':i+1,'path':str(p.relative_to(ROOT)),
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for i,p in enumerate(pages)],
            'finite_population':'kappa=2; fixed large X; p=.5; rho=.5; OLS, W0=(X,U,X-by-U), Ding; 2000 shared assignments per population',
            'superpopulation':'Observational, large X, rho=.5, cubic ridge; direct, calibrated, naive, X-only, oracle; 1000 shared datasets per sample size',
            'source_of_randomness':'Treatment assignments only within each fixed finite population.',
            'real_data_tables':'Four approved common-base X choices per dataset; Direct/Calibrated for NVSS and OLS/Model assisted for Hillstrom. Sources: real_x_comparison/.',
            'simulation_records':'Current finite figures select only OLS, model-assisted W0 and Ding from x_interactions_revision/; targeted_revision/ for superpopulation additions; earlier W comparisons remain in quadratic_x_only_revision/, targeted_revision/finite/ and finite_w_experiment/.',
            'intervals':'Wald intervals intersected with [-1,1] for rho and [0,infinity) for V11/V22; V12 unchanged. Empty intersections shown Empty; simulation coverage unchanged.',
            'pdf_pages':8}
(ROOT/'display_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Verified eight-page figures-and-tables PDF.')
