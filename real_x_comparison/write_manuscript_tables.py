#!/usr/bin/env python3
"""Write manuscript table bodies directly from full-precision estimate CSVs.

No intermediate display rounding. NVSS variance/covariance entries are multiplied
by 1000; Hillstrom visit effects use percentage-point units. Intervals are
intersected with the parameter space only for display, as in build_tables.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from build_tables import truncate_intervals

ROOT=Path(__file__).resolve().parent
TARGETS=['V11','V22','V12','rho','theta','alpha']
LABELS=[r'Base',r'Base\,+\,A',r'Base\,+\,B',r'Base\,+\,A\,+\,B']


def number(value,digits,boundary=False):
    if not np.isfinite(value):return '--'
    if boundary and value in (-1,0,1):return str(int(value))
    return f'{value:.{digits}f}'


def write(dataset,order,methods,scale,digits):
    data=pd.read_csv(ROOT/dataset/'results/estimates.csv')
    for metric in ['estimate','se','lower','upper']:
        data[metric]*=data.estimand.map(scale).fillna(1.)
    data=truncate_intervals(data)
    lines=[r'\begin{tabular}{@{}llcccccc@{}}',r'\toprule',
           r'\(\bm X\) & Method & \(V_{11}\) & \(V_{22}\) & \(V_{12}\) & \(\rho\) & \(\theta\) & \(\alpha\)\\',r'\midrule']
    for i,x in enumerate(order):
        for k,(method,name) in enumerate(methods):
            points=[];intervals=[]
            for j,target in enumerate(TARGETS):
                row=data[(data.X==x)&(data.method==method)&(data.estimand==target)].iloc[0]
                p=number(row.estimate,digits[j]);points.append('--' if p=='--' else '$'+p+'$')
                if row.interval_status=='undefined':intervals.append('')
                elif row.interval_status=='empty':intervals.append('(empty)')
                else:
                    a=number(row.lower,digits[j],True);b=number(row.upper,digits[j],True)
                    intervals.append('$('+a+', '+b+')$')
            lines.append(' & '.join([LABELS[i] if k==0 else '',name,*points])+r' \\')
            lines.append(' & '.join(['','',*intervals])+r' \\')
        if i<len(order)-1:lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    out=ROOT/'tables';out.mkdir(exist_ok=True)
    (out/(dataset+'_manuscript_table.tex')).write_text('\n'.join(lines)+'\n')


def main():
    write('nvss',['age_education','age_education_parity_bmi','age_education_smoking','age_education_parity_bmi_smoking'],
          [('direct','Direct'),('calibrated','Calibrated')],{'V11':1000.,'V22':1000.,'V12':1000.},[1,2,1,2,2,2])
    write('hillstrom',['small','large','history_newbie','combined'],[('ols','OLS'),('model_assisted','Model assisted')],
          {'ATE1':100.,'V11':10000.,'V12':100.,'theta':.01},[2,3,2,2,2,2])
    print('Wrote manuscript LaTeX table bodies from unrounded results.')


if __name__=='__main__':main()
