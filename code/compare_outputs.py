"""Compare regenerated outputs in code/ with expected_outputs/ (numeric tolerance)."""
import json, glob, os, sys, math, pandas as pd
# usage: python compare_outputs.py [TOL [EXPECTED_DIR GENERATED_DIR [FILE ...]]]
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
exp_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('..', 'expected_outputs')
gen_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
only = set(sys.argv[4:]); bad = 0
def walk(a, b, path=''):
    global bad
    if isinstance(a, dict):
        for k in a: walk(a[k], b.get(k) if isinstance(b, dict) else None, f'{path}/{k}')
    elif isinstance(a, list):
        for i, x in enumerate(a): walk(x, b[i] if isinstance(b, list) and i < len(b) else None, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if b is None or (not (isinstance(a, float) and math.isnan(a)) and abs(a - b) > TOL * max(1, abs(a))):
            bad += 1; print(f'DIFF {path}: expected {a}, got {b}')
for f in sorted(glob.glob(os.path.join(exp_dir, '*'))):
    base = os.path.basename(f)
    if only and base not in only: continue
    name = os.path.join(gen_dir, base)
    if not os.path.exists(name): print(f'MISSING {name}'); bad += 1; continue
    if name.endswith('.json'): walk(json.load(open(f)), json.load(open(name)), name)
    elif name.endswith('.csv'):
        e, g = pd.read_csv(f), pd.read_csv(name)
        num = e.select_dtypes('number').columns
        if e.shape != g.shape or ((e[num] - g[num]).abs() > TOL).any().any(): bad += 1; print(f'DIFF {name}')
print('ALL OUTPUTS MATCH' if bad == 0 else f'{bad} differences found')
sys.exit(1 if bad else 0)
