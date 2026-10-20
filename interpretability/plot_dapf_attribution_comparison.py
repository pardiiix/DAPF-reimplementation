#!/usr/bin/env python3
"""Compare already aggregated five-seed DAPF-BERT / DAPF-RoBERTa IG signs.

Outputs two plots (PNG + vector PDF) and the exact token-level matched CSV.
Plots show observed cross-backbone association, not controlled baseline-sensitivity.

Usage:
  python scripts/plot_dapf_attribution_comparison.py \
    --bert-csv output_interpretability_multiseed_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv \
    --roberta-csv output_roberta_interpretability_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv \
    --out-dir output_roberta_comparison/figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCORE = 'mean_of_seed_mean_attribution'
FILTERS = ['n_seeds', 'total_count', 'mean_n_samples_per_seed', 'token_norm', SCORE]
THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.05]


def load(path, label):
    if not path.exists():
        raise FileNotFoundError(f'{label} IG CSV not found: {path}')
    df = pd.read_csv(path)
    missing = set(FILTERS) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing required columns {sorted(missing)}')
    df = df[(df.n_seeds == 5) & (df.total_count >= 10) & (df.mean_n_samples_per_seed >= 5)]
    df = df[['token_norm', SCORE]].copy()
    df[SCORE] = pd.to_numeric(df[SCORE], errors='coerce')
    if df.token_norm.duplicated().any():
        raise ValueError(f'{path}: duplicated normalized tokens')
    return df.rename(columns={SCORE:label})


def save(fig, out, stem):
    for ext in ('png','pdf'):
        path = out/f'{stem}.{ext}'
        fig.savefig(path, dpi=220 if ext=='png' else None, bbox_inches='tight')
        print('Saved:',path)
    plt.close(fig)


def sign_scatter(df, out):
    agreed=df[df.same_sign]
    opposed=df[~df.same_sign]
    fig,ax=plt.subplots(figsize=(8.5,7))
    ax.scatter(agreed.bert,agreed.roberta,s=30,alpha=.7,label=f'Same sign (n={len(agreed)})')
    ax.scatter(opposed.bert,opposed.roberta,s=33,alpha=.75,marker='x',label=f'Opposite sign (n={len(opposed)})')
    ax.axhline(0,color='gray',linewidth=1,linestyle='--')
    ax.axvline(0,color='gray',linewidth=1,linestyle='--')
    # Annotate only the most consequential sign reversals, preventing clutter.
    annotated = opposed.nlargest(7,'combined_magnitude')
    for r in annotated.itertuples(index=False):
        ax.annotate(r.token_norm,(r.bert,r.roberta),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='DAPF-BERT: five-seed mean signed IG',
           ylabel='DAPF-RoBERTa: five-seed mean signed IG',
           title='DAPF attribution directions across backbones')
    ax.legend(loc='best',frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,out,'dapf_bert_roberta_ig_sign_scatter')


def agreement_by_threshold(df, out):
    rows=[]
    for threshold in THRESHOLDS:
        selected=df[(df.bert.abs()>threshold)&(df.roberta.abs()>threshold)]
        n=len(selected)
        same=int(selected.same_sign.sum())
        rows.append(dict(min_abs_ig_both=threshold,n_tokens=n,same=same,reversed=n-same,
                         agreement_pct=100*same/n if n else np.nan))
    res=pd.DataFrame(rows)
    res.to_csv(out/'dapf_ig_sign_agreement_thresholds.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,5))
    x=np.arange(len(res))
    ax.plot(x,res.agreement_pct,marker='o',linewidth=2)
    ax.set_xticks(x,[f'{v:g}' for v in res.min_abs_ig_both])
    ax.set_ylim(0,110)
    ax.set(xlabel='Required minimum |mean IG| in both models',
           ylabel='Same-sign tokens (%)',
           title='Cross-backbone sign agreement depends on magnitude cutoff')
    for i,r in res.iterrows():
        if np.isfinite(r.agreement_pct):
            ax.annotate(f'{r.agreement_pct:.1f}%\n(n={int(r.n_tokens)})',
                        (i,r.agreement_pct),xytext=(4,10),textcoords='offset points',fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,out,'dapf_ig_sign_agreement_thresholds')
    print('\n',res.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bert-csv',default='output_interpretability_multiseed_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv')
    p.add_argument('--roberta-csv',default='output_roberta_interpretability_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv')
    p.add_argument('--out-dir',default='output_roberta_comparison/figures')
    a=p.parse_args()
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    both=load(Path(a.bert_csv),'bert').merge(load(Path(a.roberta_csv),'roberta'),on='token_norm')
    both=both.dropna()
    both=both[(both.bert!=0)&(both.roberta!=0)].copy()
    if both.empty:
        raise ValueError('No shared nonzero tokens passed the five-seed filtering rules')
    both['same_sign']=np.sign(both.bert)==np.sign(both.roberta)
    both['combined_magnitude']=both.bert.abs()+both.roberta.abs()
    both.sort_values('combined_magnitude',ascending=False).to_csv(out/'dapf_ig_shared_tokens.csv',index=False)
    print('Shared:',len(both),'same direction:',both.same_sign.sum(),'opposite:',(~both.same_sign).sum())
    sign_scatter(both,out)
    agreement_by_threshold(both,out)

if __name__=='__main__':
    main()
