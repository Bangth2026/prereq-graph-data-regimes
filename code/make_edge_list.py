"""Regenerate edge_list/ from the Junyi release (run inside code/ after run_all.sh has extracted the release).

The public record (v1.4.4) does not redistribute the derived edge lists, because they carry exercise names
and crowd ratings from the Junyi release and the authors have not established a right to redistribute
that material. This script rebuilds them locally, with the same columns as earlier package versions:
  edge_list/fpkg_edges_crowd.csv   source,target,weight,Q,n_raters,status,layer,ratings_0to9,
                                   expert_same_direction,expert_reverse_direction   (1,131 rows)
  edge_list/fpkg_edges_expert.csv  source,target,layer,status                       (978 rows)
"""
import os, numpy as np, pandas as pd
from common import *   # names, crowd(), expert_edges()
os.makedirs('../edge_list', exist_ok=True)
cr = crowd(); Ex, removed = expert_edges()
QMIN = 0.80
rows = []
for r in cr.itertuples():
    rows.append(dict(source=names[r.src], target=names[r.dst], weight=r.mu, Q=r.Q, n_raters=r.m,
                     status='accepted' if r.Q >= QMIN - 1e-9 else 'rejected', layer='crowd', ratings_0to9=r.raw,
                     expert_same_direction=(r.src, r.dst) in Ex, expert_reverse_direction=(r.dst, r.src) in Ex))
pd.DataFrame(rows).to_csv('../edge_list/fpkg_edges_crowd.csv', index=False)
pd.DataFrame([dict(source=names[s], target=names[d], layer='expert', status='accepted (single source, unscreened)')
              for s, d in sorted(Ex)]).to_csv('../edge_list/fpkg_edges_expert.csv', index=False)
print('crowd relations', len(rows), 'accepted', sum(r['status'] == 'accepted' for r in rows), '| expert edges', len(Ex),
      '| cycle edges removed', removed)
