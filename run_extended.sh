#!/usr/bin/env bash
# Replication on ASSISTments 2009-2010, EdNet-KT1 and XES3G5M (Section 4.6, Table 8). Independent of the Junyi pipeline.
# Usage: bash run_extended.sh /path/to/skill_builder_data_corrected_collapsed.csv /path/to/kt1_0.parquet /path/to/questions.parquet /path/to/XES3G5M.tar.gz
# Approximate times on 2 CPU cores and 8 GB RAM: ASSISTments all models 40 min; EdNet DKT 20 min, SAKT 50 min, AKT 75 min;
# XES3G5M DKT 3 min, SAKT 20 min, AKT 55 min, the two evaluations 15 min.
# The cross-dataset summary (replication_summary.py) also needs code/out_benchmark.json from run_all.sh.
set -euo pipefail
AS="$(realpath "$1")"; KT1="$(realpath "$2")"; QS="$(realpath "$3")"; XT="$(realpath "$4")"
cd "$(dirname "$0")/code"
python verify_extra.py "$AS" "$KT1" "$QS" "$XT"
A=xds/assist09; E=xds/ednet; mkdir -p $A $E
python xds_prep.py assist09 "$AS" $A
python xds_prep.py ednet "$KT1" "$QS" $E
for D in $A $E; do python xds_eval.py queries $D; done
NA=$(($(wc -l < $A/units.csv)-1)); NE=$(($(wc -l < $E/units.csv)-1))
python kt_dkt.py 10 $A/sorted.npz $NA $A
python kt_attn.py sakt $A/sorted.npz $NA $A/kt_queries.npz $A/kt_sakt.npz 5 1.0
python kt_attn.py akt  $A/sorted.npz $NA $A/kt_queries.npz $A/kt_akt.npz  5 1.0
python kt_dkt.py 2 $E/sorted.npz $NE $E
python kt_attn.py sakt $E/sorted.npz $NE $E/kt_queries.npz $E/kt_sakt.npz 1 1.0
python kt_attn.py akt  $E/sorted.npz $NE $E/kt_queries.npz $E/kt_akt.npz  1 0.5
for D in $A $E; do python xds_eval.py eval $D; done
for D in assist09 ednet; do python compare_outputs.py 1e-3 ../replication/$D xds/$D prep.json relations.csv out_xds.json; done
# XES3G5M: curriculum-tree relations (relations.csv) and, for comparison, mined relations (relations_mined.csv)
X=xds/xes3g5m; XM=xds/xes3g5m_mined; mkdir -p xds/raw $X $XM; tar -xzf "$XT" -C xds/raw
python xds_prep.py xes xds/raw/XES3G5M $X
for f in sorted.npz units.csv prep.json; do ln -sf ../xes3g5m/$f $XM/$f; done; cp $X/relations_mined.csv $XM/relations.csv
python xds_eval.py queries $X; python xds_eval.py queries $XM
python -c "import numpy as np; a=np.load('$X/kt_queries.npz'); b=np.load('$XM/kt_queries.npz'); k=np.union1d(a['rows'].astype(np.int64)*4096+a['ex'], b['rows'].astype(np.int64)*4096+b['ex']); np.savez('$X/kt_queries_union.npz', rows=k//4096, ex=k%4096)"
NX=$(($(wc -l < $X/units.csv)-1))
python kt_dkt.py 10 $X/sorted.npz $NX $X
python kt_attn.py sakt $X/sorted.npz $NX $X/kt_queries_union.npz $X/kt_sakt.npz 1 1.0
python kt_attn.py akt  $X/sorted.npz $NX $X/kt_queries_union.npz $X/kt_akt.npz  1 0.5
for f in dkt_states.npz kt_sakt.npz kt_akt.npz; do ln -sf ../xes3g5m/$f $XM/$f; done
python xds_eval.py eval $X; python xds_eval.py eval $XM
python compare_outputs.py 1e-3 ../replication/xes3g5m $X prep.json relations.csv relations_mined.csv out_xds.json
python compare_outputs.py 1e-3 ../replication/xes3g5m_mined $XM relations.csv out_xds.json
# KT training length on ASSISTments and XES3G5M: DKT for other numbers of epochs and to early stopping,
# learner models only, graph evidence over B1RK with DKT (FORCE_KT); about 1.5 h
for spec in $A:2:0 $A:80:3 $X:1:0 $X:3:0 $X:40:2; do
  IFS=: read B ep pat <<< "$spec"; D=$B/curve_e$ep; mkdir -p $D
  for f in sorted.npz units.csv relations.csv prep.json kt_sakt.npz kt_akt.npz; do ln -sf ../$f $D/$f; done
  NU=$(($(wc -l < $B/units.csv)-1))
  if [ "$pat" = "0" ]; then python kt_dkt.py $ep $B/sorted.npz $NU $D; else python kt_dkt.py $ep $B/sorted.npz $NU $D $pat; fi
  python xds_eval.py eval $D
done
for D in $A $X $A/curve_e2 $A/curve_e80 $X/curve_e1 $X/curve_e3 $X/curve_e40; do ONLY_LM=1 FORCE_KT=DKT python xds_eval.py eval $D; done
if [ -f out_benchmark.json ]; then python replication_summary.py $A $E $X $XM; python compare_outputs.py 1e-3 ../replication . out_summary.json; fi
