"""Three superpopulation displays including paired X-only adjustment."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

OUT = Path(__file__).resolve().parent
SP = OUT / 'source_data/super_confirm_summary.csv'
FP = OUT / 'source_data/finite_v2_summary.csv'
TARGETS = ['V11','V22','V12','rho','theta','alpha']
LABELS = [r'$V_{11}$',r'$V_{22}$',r'$V_{12}$',r'$\rho$',r'$\theta$',r'$\alpha$']
COLORS = {'direct':'#2675AA','calibrated':'#D55E00','naive':'#318567','oracle':'#777777','x_only':'#9467BD',
          'ols':'#2675AA','model_assisted':'#D55E00'}
METHODS = {'direct':'Direct','calibrated':'Calibrated','naive':'Naïve','oracle':'Oracle EIF','x_only':'X-only',
           'ols':'OLS','model_assisted':'Model assisted'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.3,
 'axes.titlesize':12.5,'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B0B4BA',
 'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white',
 'savefig.facecolor':'white'})

S0 = pd.read_csv(SP)
S = S0[(S0.design=='observational') & (S0.nested=='large') &
       S0.method.isin(['direct','calibrated','naive','x_only','oracle']) & S0.estimand.isin(TARGETS)].copy()
assert len(S)==150
assert not S.duplicated(['n','method','estimand']).any()
assert set(S.attempted)=={1000}
assert set(S.learner)=={'cubic_ridge'} and set(S.rho)=={.5}
assert np.isfinite(S[['bias','rmse','coverage']]).all().all()
S.to_csv(OUT/'superpopulation_plot_data.csv',index=False)

def fmt_x(value, pos):
    return f'{value/1000:g}k' if value>=1000 else f'{int(value)}'

def draw(framework, metric, data):
    is_super = framework=='superpopulation'
    title = 'Superpopulation' if is_super else 'Finite population'
    mtitle = {'coverage':'95% interval coverage','bias':'Bias','rmse':'RMSE'}[metric]
    fig, axes = plt.subplots(2,3,figsize=(12.0,6.5))
    fig.subplots_adjust(left=.075,right=.985,bottom=.10,top=.795,wspace=.25,hspace=.42)
    fig.suptitle(f'{title}: {mtitle}',fontsize=17.5,fontweight='bold',y=.975,x=.075,ha='left')
    ngrid=sorted(data.n.unique())
    if is_super:
        series=[(m,None) for m in ['direct','calibrated','naive','x_only','oracle']]
        handles=[Line2D([0],[0],color=COLORS[m],lw=2,marker='o',markersize=4,
                       linestyle='--' if m=='oracle' else '-',label=METHODS[m]) for m,_ in series]
    else:
        series=[(m,k) for k in [0,2] for m in ['ols','model_assisted']]
        handles=[Line2D([0],[0],color=COLORS[m],lw=2,marker='o' if k==0 else 's',markersize=4,
                       linestyle='-' if k==0 else '--',label=METHODS[m]+rf', $\kappa={k}$') for m,k in series]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.067,.894),frameon=False,
               ncol=5,columnspacing=1.7,handlelength=2.5,fontsize=10.7)
    for ax,target,lab in zip(axes.flat,TARGETS,LABELS):
        d=data[data.estimand==target]
        truth=d.truth.iloc[0]
        assert np.allclose(d.truth,truth)
        ax.set_title(lab,pad=9)
        for method,kappa in series:
            q=d[d.method==method]
            if kappa is not None: q=q[q.kappa==kappa]
            q=q.sort_values('n')
            assert q.n.tolist()==ngrid
            ax.plot(q.n,q[metric],color=COLORS[method],lw=1.9,
                    marker='s' if kappa==2 else 'o',markersize=4.5,
                    linestyle='--' if method=='oracle' or kappa==2 else '-',
                    zorder=3,markeredgewidth=.7,markeredgecolor='white')
        ax.set_xscale('log')
        ax.set_xticks(ngrid)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='minor',bottom=False)
        ax.set_xlim(ngrid[0]/1.17,ngrid[-1]*1.17)
        ax.grid(axis='y',color='#E8EBEF',lw=.6,zorder=0)
        ax.set_axisbelow(True)
        if metric=='coverage':
            ax.axhline(.95,color='#4B515A',lw=.85,ls=':',zorder=2)
            ax.set_ylim((0,1.01) if is_super else (.85,1.005))
            ax.set_yticks([0,.25,.5,.75,.95] if is_super else [.85,.9,.95,1])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.0%}'))
        elif metric=='bias':
            ax.axhline(0,color='#777D84',lw=.9,ls=':',zorder=2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        else:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=8))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if ax in axes[1]: ax.set_xlabel('Sample size n (log scale)')
    for ax in axes[:,0]:
        ax.set_ylabel({'coverage':'Coverage','bias':'Mean estimate − truth','rmse':'RMSE (log scale)'}[metric])
    path=OUT/f'{framework}_{metric}'
    buffer = BytesIO()
    fig.savefig(buffer, format='pdf')
    payload = buffer.getvalue()
    assert payload.rstrip().endswith(b'%%EOF')
    temporary = path.with_suffix('.pdf.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'),dpi=180)
    plt.close(fig)

for metric in ['coverage','bias','rmse']:
    draw('superpopulation',metric,S)
