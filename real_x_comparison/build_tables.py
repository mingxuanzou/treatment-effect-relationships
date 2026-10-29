from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'tables'
OUT.mkdir(exist_ok=True)
TARGETS = ['ATE1','ATE2','V11','V22','V12','rho','theta','alpha']
LABELS = [r'$\bar\tau_1$',r'$\bar\tau_2$',r'$V_{11}$',r'$V_{22}$',r'$V_{12}$',r'$\rho$',r'$\theta$',r'$\alpha$']
METHODS = {'naive':'Naïve','direct':'Direct','calibrated':'Calibrated','ols':'OLS','model_assisted':'Model assisted'}
NAVY='#182B49'; INK='#1B2838'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':10})

def fmt(x, digits=None, boundary=False):
    if not np.isfinite(x): return 'Undefined'
    if boundary and x in (-1,0,1): return str(int(x))
    return (f'{x:.4g}' if digits is None else f'{x:.{int(digits)}f}').replace('-', '−')

def atom(path, payload):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(payload)
    tmp.replace(path)

def truncate_intervals(df):
    """Intersect Wald intervals with known marginal parameter spaces.

    Raw estimation results remain untouched. Empty intersections are reported
    explicitly rather than replaced by a boundary point.
    """
    df=df.copy()
    df['wald_lower']=df.lower
    df['wald_upper']=df.upper
    finite=np.isfinite(df.lower)&np.isfinite(df.upper)
    df['interval_status']=np.where(finite,'nonempty','undefined')
    variance=df.estimand.isin(['V11','V22'])&finite
    correlation=df.estimand.eq('rho')&finite
    df.loc[variance,'lower']=np.maximum(0.,df.loc[variance,'lower'])
    df.loc[correlation,'lower']=np.maximum(-1.,df.loc[correlation,'lower'])
    df.loc[correlation,'upper']=np.minimum(1.,df.loc[correlation,'upper'])
    empty=finite&(df.lower>df.upper)
    df.loc[empty,'interval_status']='empty'
    df.loc[empty,['lower','upper']]=np.nan
    return df


def interval_cell(d):
    if not np.isfinite(d.estimate): return 'Undefined'
    digits = d.get('display_digits', None)
    if d.interval_status=='empty': return f'{fmt(d.estimate, digits)}\n(Empty)'
    if d.interval_status=='undefined': return f'{fmt(d.estimate, digits)}\n(Undefined)'
    return f'{fmt(d.estimate, digits)}\n({fmt(d.lower, digits, True)}, {fmt(d.upper, digits, True)})'


def table(ax, rows, labels, widths, blocks, size=10):
    ax.axis('off')
    t=ax.table(cellText=rows,colLabels=labels,colWidths=widths,cellLoc='center',colLoc='center',bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(size)
    for (r,c),cell in t.get_celld().items():
        cell.set_linewidth(.45); cell.set_edgecolor('#DCE2E8')
        if r==0:
            cell.set_facecolor('#E9EEF5');cell.set_text_props(color=NAVY,weight='bold',fontsize=size+1)
        else:
            cell.set_facecolor('#F5F8FC' if blocks[r-1]%2 else 'white')
            cell.set_text_props(color=INK,weight='bold' if c==0 else 'normal')
    return t

def estimates_table(fig, df, xorder, names, methods, rect, size=10, labels=None):
    rows=[]; blocks=[]
    for b,x in enumerate(xorder):
        for im,m in enumerate(methods):
            row=[names[x] if im==0 else '',METHODS[m]]
            for target in TARGETS:
                d=df[(df.X==x)&(df.method==m)&(df.estimand==target)].iloc[0]
                row.append(interval_cell(d))
            rows.append(row);blocks.append(b)
    return table(fig.add_axes(rect),rows,['X','Method']+(LABELS if labels is None else labels),[.23,.105]+[.083125]*8,blocks,size)

def nvss():
    d=pd.read_csv(ROOT/'nvss/results/estimates.csv')
    d=d[d.method.isin(['direct','calibrated'])].copy()
    for col in ['estimate','se','lower','upper']:
        d.loc[d.estimand.isin(['V11','V22','V12']), col] *= 1000.
    d['display_digits']=d.estimand.map({'ATE1':2,'ATE2':2,'V11':1,'V22':2,'V12':1,'rho':2,'theta':2,'alpha':2})
    d=truncate_intervals(d)
    order=['age_education','age_education_parity_bmi','age_education_smoking','age_education_parity_bmi_smoking']
    names={order[0]:'Age, education',order[1]:'Age, education, prior births,\nprepregnancy BMI',order[2]:'Age, education,\nprepregnancy smoking dose',order[3]:'Age, education, prior births,\nprepregnancy BMI, smoking dose'}
    assert set(d.X)==set(order) and len(d)==64
    fig=plt.figure(figsize=(20.5,7.8),facecolor='white')
    fig.text(.035,.965,'NVSS · Superpopulation',fontsize=20,weight='bold',color=NAVY,va='top')
    estimates_table(fig,d,order,names,['direct','calibrated'],[.035,.065,.93,.825],10.5,labels=LABELS[:2]+[r'$10^3 V_{11}$',r'$10^3 V_{22}$',r'$10^3 V_{12}$']+LABELS[5:])
    d.to_csv(OUT/'nvss_display_data.csv',index=False)
    return fig

def hillstrom():
    d=pd.read_csv(ROOT/'hillstrom/results/estimates.csv')
    order=['small','large','history_newbie','combined']
    assert set(d.X)==set(order) and len(d)==64
    aliases={
        'small':'Recency, prior spending',
        'large':'Recency, prior spending,\nmen’s/women’s purchases',
        'category_only':'Men’s/women’s purchases',
        'categories':'Men’s/women’s purchases',
        'categories_only':'Men’s/women’s purchases',
        'history_newbie':'Recency, prior spending,\nnew-customer status',
        'history_new_customer':'Recency, prior spending,\nnew-customer status',
        'small_newbie':'Recency, prior spending,\nnew-customer status',
        'combined':'Recency, prior spending, purchase\ncategories, new-customer status',
    }
    unknown=set(order)-set(aliases)
    if unknown: raise ValueError(f'Add explicit X label for {unknown}')
    scales={'ATE1':100.,'ATE2':1.,'V11':10000.,'V22':1.,'V12':100.,'rho':1.,'theta':.01,'alpha':1.}
    for col in ['estimate','se','lower','upper']:
        d[col] *= d.estimand.map(scales)
    d['display_digits']=d.estimand.map({'ATE1':2,'ATE2':2,'V11':2,'V22':3,'V12':2,'rho':2,'theta':2,'alpha':2})
    d=truncate_intervals(d)
    fig=plt.figure(figsize=(19,7.8),facecolor='white')
    fig.text(.035,.965,'Hillstrom · Finite population',fontsize=20,weight='bold',color=NAVY,va='top')
    estimates_table(fig,d,order,aliases,['ols','model_assisted'],[.035,.065,.93,.825],10.5)
    d.to_csv(OUT/'hillstrom_display_data.csv',index=False)
    return fig

def main():
    figures=[('nvss_x_comparison',nvss()),('hillstrom_x_comparison',hillstrom())]
    pdfbytes=BytesIO()
    with PdfPages(pdfbytes) as pdf:
        for name,fig in figures:
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            for ax in fig.axes:
                for tbl in ax.tables:
                    for cell in tbl.get_celld().values():
                        text=cell.get_text()
                        box=text.get_window_extent(renderer)
                        available=cell.get_window_extent(renderer)
                        if box.width>0:
                            ratio=min(1.,.92*available.width/box.width,.85*available.height/box.height)
                            text.set_fontsize(text.get_fontsize()*ratio)
                            assert text.get_fontsize()>=8., (name,text.get_text(),text.get_fontsize())
            fig.canvas.draw()
            png=BytesIO();fig.savefig(png,format='png',dpi=160,bbox_inches='tight',pad_inches=.18)
            atom(OUT/(name+'.png'),png.getvalue())
            with Image.open(OUT/(name+'.png')) as im: im.verify()
            pdf.savefig(fig,bbox_inches='tight',pad_inches=.18)
            plt.close(fig)
    payload=pdfbytes.getvalue();assert payload.rstrip().endswith(b'%%EOF')
    atom(ROOT/'real_data_X_comparison.pdf',payload)
    reader=PdfReader(ROOT/'real_data_X_comparison.pdf');assert len(reader.pages)==2
    for (name,_),page in zip(figures,reader.pages):
        writer=PdfWriter();writer.add_page(page);pagebuffer=BytesIO();writer.write(pagebuffer)
        atom(OUT/(name+'.pdf'),pagebuffer.getvalue())
    text=''.join(p.extract_text().lower() for p in reader.pages)
    assert 'eigen' not in text and 'naïve' not in text and 'naive' not in text
    assert 'choices of x' not in text and 'marginal variance estimates' not in text
    (OUT/'display_validation.json').write_text(json.dumps({'pdf_pages':2,'tables':2,'X_choices_per_dataset':4,'design':'common base; add A; add B; add both','nvss_methods':['direct','calibrated'],'hillstrom_methods':['ols','model_assisted'],'PNG_integrity':'passed','eigenvalues_displayed':False,'Hillstrom_Y1_display':'percentage points','intervals':'95% Wald intersected with [-1,1] for rho and [0,infinity) for V11/V22; other intervals unchanged; disjoint intersections reported Empty','figures_modified':False,'separate_marginal_variance_section':False,'choices_of_X_title_text':False},indent=2))
    print('Rendered two X-comparison tables.')

if __name__=='__main__': main()
