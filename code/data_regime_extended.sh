#!/usr/bin/env bash
# Extended data-regime study on Junyi (post hoc): adds a 3.2% size (about 4,000 learners), and for every point runs
# data_regime_point.py twice: validation learners kept whole, and validation learners subsampled in the same proportion (VALSUB=1).
cd "$(dirname "$0")"
for F in 0.5 2 3.2 5 1 10 25 100; do
for S in 1 2 3; do
  RD=data/regime/f${F}_s${S}; mkdir -p $RD
  [ -f $RD/out2.json ] && [ -f $RD/out2_vs.json ] && continue
  if [ ! -f $RD/learners.npz ]; then python -c "
import numpy as np; z=np.load('data/sorted.npz'); lid=z['lid']; n=z['n']
st=np.r_[0,np.flatnonzero(lid[1:]!=lid[:-1])+1]; L=lid[st][n[st]>=10]
k=max(30,int(round($F/100*len(L)))); rng=np.random.default_rng($S)
np.savez('$RD/learners.npz', L=np.sort(rng.choice(L,k,replace=False)) if k<len(L) else L)"; fi
  DKT_SUBSET=$RD/learners.npz DKT_SEED=$S python kt_dkt.py 30 data/sorted.npz 837 $RD 2 > $RD/dkt2.log 2>&1
  [ -f $RD/out.json ] || python graph_estimator.py regime $F $S > $RD/gpe.log 2>&1
  [ -f $RD/out2.json ] || python data_regime_point.py regime $F $S > $RD/gpe2.log 2>&1
  [ -f $RD/out2_vs.json ] || VALSUB=1 python data_regime_point.py regime $F $S > $RD/gpe2_vs.log 2>&1
  rm -f $RD/dkt_states.npz
  echo "$F $S $(date +%H:%M)" >> data/regime/orch2.log
done; done
echo done > data/regime/ALLDONE2
