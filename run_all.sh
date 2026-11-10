#!/usr/bin/env bash
# Full reproduction. Usage: bash run_all.sh /path/to/junyi.rar
set -euo pipefail
RAR="$(realpath "${1:-junyi.rar}")"
cd "$(dirname "$0")/code"
mkdir -p data fig
python verify_download.py "$RAR"
bsdtar -xf "$RAR" -C data relationship_annotation_training.csv relationship_annotation_testing.csv junyi_Exercise_table.csv README.txt
bsdtar -xOf "$RAR" junyi_ProblemLog_original.csv | python conv.py
python prep.py
python repro.py
python make_edge_list.py
python build.py 0.6 0.8 val
python build.py 0.8 1.01 test
python build.py 0.7 1.01 t70
python build.py 0.9 1.01 t90
python run_main.py new
python run_forms.py
python run_graph.py
python run_rank.py
python run_bkt.py
python run_stab.py
python run_sens.py
python run_diag.py
python run_explor.py
python run_cases.py
python analysis_a.py
python analysis_b.py
python analysis_c.py
python worked_example.py
python analysis_e.py
python analysis_f.py
python figs.py
python figs2.py
# estimators, ESD, protocols, stronger learner models; needs torch (CPU)
python bkt_prior.py
python kt_dkt.py 2
# SAKT and AKT (about 2 h each on 2 CPU cores), then the benchmark that uses DKT, SAKT and AKT
python kt_queries.py
python kt_attn.py sakt data/sorted.npz 837 data/kt_queries.npz data/kt_sakt.npz 1 1.0
python kt_attn.py akt  data/sorted.npz 837 data/kt_queries.npz data/kt_akt.npz  1 0.35
python benchmark.py
python agreement.py
# controls: ablation of the decomposition, random-graph control, calibration, Table 6 failures flagged (about 15 min)
python controls.py
python table6_flags.py
# KT training length: DKT for 1 and 5 epochs and to early stopping (about 30 min), then the graph increment over each
mkdir -p data/dkt_e1 data/dkt_e5 data/dkt_conv
python kt_dkt.py 1 data/sorted.npz 837 data/dkt_e1
python kt_dkt.py 5 data/sorted.npz 837 data/dkt_e5
python kt_dkt.py 30 data/sorted.npz 837 data/dkt_conv 2
for D in data/dkt_e1 data data/dkt_e5 data/dkt_conv; do python kt_training_length.py $D; done
python teacher_card.py
python latency.py      # machine-dependent timings; output kept in supplementary_outputs/, not compared
python latency_estimators.py
python figures_benchmark.py
python compare_outputs.py
