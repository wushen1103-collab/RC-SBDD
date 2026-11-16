"""Offline checks of frozen selection, risk inference, and calibrated decisions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parent
TAU=0.32490503083644007

def select(group,policy,k):
    g=group[group.intramol_pass].copy()
    if policy=='PB-QED':return g.sort_values(['qed','risk_prob'],ascending=[False,True]).head(k)
    safe=g[g.risk_prob.le(TAU)].sort_values(['qed','risk_prob'],ascending=[False,True]).head(k)
    if len(safe)>=k:return safe
    rest=g.drop(safe.index).assign(weighted=lambda x:x.qed-x.risk_prob)
    fill=rest.sort_values(['weighted','qed'],ascending=[False,False]).head(k-len(safe))
    return pd.concat([safe,fill])

def main():
    expected=json.loads((ROOT/'SHA256.json').read_text())
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    output=[]
    pools={}
    for generator,k in [('DiffSBDD',10),('Pocket2Mol',4)]:
        pool=pd.read_csv(ROOT/'data'/f'{generator}.csv');pools[generator]=pool
        for policy in ['PB-QED','PB-RC']:
            s=pd.concat([select(g,policy,k) for _,g in pool.groupby('key')])
            a=s.groupby('key').agg(dock_fast=('dock_pose_pass','mean'),qed=('qed','mean'))
            output.append(dict(generator=generator,policy=policy,k=k,targets=len(a),n=len(s),pass_rate=float(a.dock_fast.mean()),qed=float(a.qed.mean())))
    primary=pd.DataFrame(output)
    assert np.allclose(primary.pass_rate,[.798,.966,.8160377358490566,.9669811320754716])
    params=json.loads((ROOT/'model/logistic_parameters.json').read_text())
    x=pd.read_csv(ROOT/'data/DiffSBDD_features.csv')
    z=((x[params['feature_names']].to_numpy()-params['mean'])/params['scale'])@np.array(params['coef'])+params['intercept']
    scores=1/(1+np.exp(-z))
    compare=x[['key','mol_index']].assign(recomputed=scores).merge(pools['DiffSBDD'][['key','mol_index','risk_prob']],on=['key','mol_index'],validate='one_to_one')
    score_error=float(np.max(np.abs(compare.recomputed-compare.risk_prob)))
    assert score_error<1e-5,score_error
    calrows=[]
    for generator,pool in pools.items():
        for seed in range(2026,2031):
            ts=np.array(sorted(pool.key.unique()));np.random.default_rng(seed).shuffle(ts)
            nc=max(5,int(round(.4*len(ts))));cal=pool[pool.key.isin(ts[:nc])];test=pool[pool.key.isin(ts[nc:])].copy()
            iso=IsotonicRegression(out_of_bounds='clip').fit(cal.risk_prob,1-cal.dock_pose_pass.astype(float))
            test['failure_prob']=iso.predict(test.risk_prob)
            tau=float(cal[cal.dock_pose_pass].risk_prob.quantile(.95))
            for method in ['raw_target_calibrated_tau','isotonic_probability_alpha']:
                chosen=[]
                for _,g in test.groupby('key'):
                    g=g[g.intramol_pass]
                    g=g[g.risk_prob.le(tau)] if method.startswith('raw') else g[g.failure_prob.le(.1)]
                    chosen.append(g.sort_values(['qed','risk_prob'],ascending=[False,True]).head(10))
                s=pd.concat(chosen)
                calrows.append(dict(generator=generator,seed=seed,method=method,coverage=s.key.nunique()/len(ts[nc:]),failure=1-s.dock_pose_pass.mean()))
    a=pd.DataFrame(calrows).sort_values(['generator','seed','method']).reset_index(drop=True)
    b=pd.read_csv(ROOT/'data/calibrated_decision_raw.csv').sort_values(['generator','seed','method']).reset_index(drop=True)
    assert np.allclose(a[['coverage','failure']],b[['coverage','failure']])
    cert=pd.read_csv(ROOT/'data/top1_certification_raw.csv')
    (ROOT/'REPRODUCTION_RESULT.json').write_text(json.dumps({'status':'passed','primary_selection':output,'maximum_risk_score_error':score_error,'calibrated_decision_rows_verified':len(a),'archived_certification_rows':len(cert)},indent=2),encoding='utf-8')
    print(primary.to_string(index=False));print('PASS: hashes, frozen risk inference, principal selection, and 20 calibrated-decision split/policy records.')

if __name__=='__main__':main()
