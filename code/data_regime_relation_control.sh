#!/usr/bin/env bash
# Relation-specific part of the graph's gain (Junyi; post hoc): real minus topic- and observation-matched random relations,
# over B1RK and over the strongest learner model, with learner-bootstrap replicates. Every layer subsampled (VALSUB=1) at
# every size; logistic layers on all validation learners at 0.5, 2 and 5%.
cd "$(dirname "$0")"
for F in 0.5 2 5 1 3.2 10 25 100; do
for S in 1 2 3; do
  RD=data/regime/f${F}_s${S}
  NEEDW=0; case $F in 0.5|2|5) NEEDW=1;; esac
  [ -f $RD/out3_vs.json ] && { [ $NEEDW = 0 ] || [ -f $RD/out3.json ]; } && continue
  DKT_SUBSET=$RD/learners.npz DKT_SEED=$S python kt_dkt.py 30 data/sorted.npz 837 $RD 2 > $RD/dkt3.log 2>&1
  [ -f $RD/out3_vs.json ] || VALSUB=1 python data_regime_relation_control.py regime $F $S > $RD/gpe3_vs.log 2>&1
  [ $NEEDW = 1 ] && { [ -f $RD/out3.json ] || python data_regime_relation_control.py regime $F $S > $RD/gpe3.log 2>&1; }
  rm -f $RD/dkt_states.npz
  echo "$F $S $(date +%H:%M)" >> data/regime/orch3.log
done; done
echo done > data/regime/ALLDONE3
