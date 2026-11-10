#!/usr/bin/env bash
# Data-regime study on XES3G5M (curriculum-tree relations); starts after the Junyi study.
cd "$(dirname "$0")"

X=${1:?path to the prepared XES3G5M folder}

for S in 1 2 3; do
for F in 1 5 25 100; do
  RD=$X/regime/f${F}_s${S}; mkdir -p $RD
  [ -f $RD/out.json ] && continue
  python -c "
import numpy as np; z=np.load('$X/sorted.npz'); lid=z['lid']; n=z['n']
st=np.r_[0,np.flatnonzero(lid[1:]!=lid[:-1])+1]; L=lid[st][n[st]>=10]
k=max(30,int(round($F/100*len(L)))); rng=np.random.default_rng($S)
np.savez('$RD/learners.npz', L=np.sort(rng.choice(L,k,replace=False)) if k<len(L) else L)"
  DKT_SUBSET=$RD/learners.npz DKT_SEED=$S python kt_dkt.py 40 $X/sorted.npz 798 $RD 2 > $RD/dkt.log 2>&1
  python graph_estimator_replication.py regime $X $F $S > $RD/gpe.log 2>&1
  rm -f $RD/dkt_states.npz
done; done
echo done > $X/regime/ALLDONE
