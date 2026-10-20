#!/usr/bin/env python3
"""Plot five-seed RoBERTa faithfulness deletion and insertion curves.

Consumes existing per-participant curve CSVs; no model inference or reruns.
One PNG and one vector PDF per ranking × perturbation method (4 figures).

Usage:
  python scripts/plot_roberta_faithfulness_comparison.py \
    --dapf-root output_roberta_interpretability \
    --cls-root output_roberta_cls_interpretability_multiseed \
    --prompt-root output_roberta_cls_prompt_interpretability_multiseed \
    --out-dir output_roberta_comparison/figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('DAPF-RoBERTa','RoBERTa-CLS','RoBERTa-CLS+Prompt')
RANKINGS = ('positive','absolute')
METHODS = {'deletion': 'probability_drop','insertion': 'probability_recovered'}


def save(fig, out, stem):
    for ext in ('png','pdf'):
        p=out/f'{stem}.{ext}'
        fig.savefig(p,dpi=220 if ext=='png' else None,bbox_inches='tight')
        print('Saved:',p)
    plt.close(fig)


def load_files(root, model):
    rows=[]
    for seed in range(5):
        for ranking in RANKINGS:
            for method, metric in METHODS.items():
                path=root/f'seed_{seed}'/f'faithfulness_{ranking}'/f'{method}_curve.csv'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {model} seed {seed}: {path}')
                df=pd.read_csv(path)
                missing={'fraction',metric,'id'}-set(df.columns)
                if missing:
                    raise ValueError(f'{path}: missing columns {sorted(missing)}')
                df=df[['id','fraction',metric]].copy()
                df['fraction']=pd.to_numeric(df.fraction,errors='coerce')
                df[metric]=pd.to_numeric(df[metric],errors='coerce')
                if df[['fraction',metric]].isna().any().any():
                    raise ValueError(f'NaN/non-numeric value in {path}')
                # Crucial: participants -> one mean per seed, then mean±SD across seeds.
                stats=df.groupby('fraction')[metric].agg(['mean','count']).reset_index()
                if not (stats['count']==48).all():
                    raise ValueError(f'{path}: expected 48 participants at each fraction: {stats.to_dict("records")}')
                for x in stats.itertuples():
                    rows.append({'model':model,'seed':seed,'ranking':ranking,'method':method,
                                 'fraction':x.fraction,'mean_for_seed':x.mean,'n_participants':x.count})
                print(f'{model}, seed {seed}, {ranking}/{method}: {len(stats)} fractions')
    return pd.DataFrame(rows)


def plot_one(df, ranking, method, out):
    fig,ax=plt.subplots(figsize=(8.7,5.5))
    for model in MODELS:
        sub=df[(df.model==model)&(df.ranking==ranking)&(df.method==method)]
        grp=sub.groupby('fraction').mean_for_seed.agg(['mean','std','count']).sort_index()
        if not (grp['count']==5).all():
            raise ValueError(f'Incomplete five-seed values for {model}/{ranking}/{method}')
        x=grp.index.to_numpy()*100
        y=grp['mean'].to_numpy()
        sd=grp['std'].to_numpy()
        line,=ax.plot(x,y,marker='o',linewidth=1.9,label=model)
        ax.fill_between(x,y-sd,y+sd,color=line.get_color(),alpha=.14)
    ax.axhline(0,color='gray',linestyle=':',linewidth=1)
    ax.set(xlabel='Perturbed fraction of ranked transcript tokens (%)',
           ylabel='Original − deleted P(dementia)' if method=='deletion' else 'Restored − baseline P(dementia)',
           title=f'RoBERTa {method}: {ranking} IG ranking (5-seed mean ± SD)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig,out,f'roberta_faithfulness_{ranking}_{method}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dapf-root',default='output_roberta_interpretability')
    p.add_argument('--cls-root',default='output_roberta_cls_interpretability_multiseed')
    p.add_argument('--prompt-root',default='output_roberta_cls_prompt_interpretability_multiseed')
    p.add_argument('--out-dir',default='output_roberta_comparison/figures')
    a=p.parse_args()
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    roots=(a.dapf_root,a.cls_root,a.prompt_root)
    df=pd.concat([load_files(Path(root),model) for root,model in zip(roots,MODELS)],ignore_index=True)
    df.to_csv(out/'roberta_faithfulness_per_seed.csv',index=False)
    grouped=df.groupby(['model','ranking','method','fraction']).mean_for_seed.agg(
        mean_across_seeds='mean',sd_across_seeds='std',n_seeds='count').reset_index()
    grouped.to_csv(out/'roberta_faithfulness_mean_sd.csv',index=False)
    for ranking in RANKINGS:
        for method in METHODS:
            plot_one(df,ranking,method,out)
    print('COMPLETE: five-seed faithfulness figures and CSVs under',out)

if __name__=='__main__':
    main()
