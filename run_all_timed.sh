#!/usr/bin/env bash
# Same steps as run_all.sh, each timed; usage: bash run_all_timed.sh /path/to/junyi.rar
set -euo pipefail
RAR="$(realpath "${1:-junyi.rar}")"
cd "$(dirname "$0")/code"
mkdir -p data fig
T0=$(date +%s)
step(){ local name="$1"; shift; local s=$(date +%s); echo "=== [$(date -u +%FT%TZ)] START $name"; "$@"; local e=$(date +%s); echo "=== END $name  wall=$((e-s))s"; }
step verify_download python verify_download.py "$RAR"
step extract bsdtar -xf "$RAR" -C data relationship_annotation_training.csv relationship_annotation_testing.csv junyi_Exercise_table.csv README.txt
step conv bash -c "bsdtar -xOf '$RAR' junyi_ProblemLog_original.csv | python conv.py"
step prep python prep.py
step repro python repro.py make_edge_list.py
step build_val python build.py 0.6 0.8 val
step build_test python build.py 0.8 1.01 test
step build_t70 python build.py 0.7 1.01 t70
step build_t90 python build.py 0.9 1.01 t90
for s in "run_main.py new" run_forms.py run_graph.py run_rank.py run_bkt.py run_stab.py run_sens.py run_diag.py run_explor.py run_cases.py analysis_a.py analysis_b.py analysis_c.py worked_example.py analysis_e.py analysis_f.py figs.py figs2.py compare_outputs.py; do
  step "$s" python $s
done
echo "=== TOTAL wall=$(( $(date +%s)-T0 ))s"
