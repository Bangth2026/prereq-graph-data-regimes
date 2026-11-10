import pandas as pd, numpy as np, networkx as nx
ex = pd.read_csv('data/junyi_Exercise_table.csv', encoding='utf-8-sig')
names=list(ex.name); idx={n:i for i,n in enumerate(names)}; NC=len(names)
area=ex.area.values
def expert_edges():
    E=set()
    for _,r in ex.iterrows():
        if isinstance(r.prerequisites,str):
            for p in r.prerequisites.split(','):
                p=p.strip()
                if p and p!=r['name']: E.add((idx[p],idx[r['name']]))
    G=nx.DiGraph(); G.add_edges_from(E); removed=[]
    while True:
        try: cyc=nx.find_cycle(G)
        except nx.NetworkXNoCycle: break
        G.remove_edge(*cyc[-1]); removed.append(cyc[-1])
    return set(G.edges()), removed
def crowd(file='data/relationship_annotation_training.csv'):
    d=pd.read_csv(file)
    rows=[]
    for _,r in d.iterrows():
        x=np.array([int(v) for v in r.Prerequisite_raw.split('_')])/9.0
        med=np.median(x); m=len(x)
        Q=1-np.abs(x-med).sum()/(m-1)
        # crowd directed edge: B -> A (B is prerequisite of A); see direction check
        rows.append(dict(src=idx[r.Exercise_B],dst=idx[r.Exercise_A],mu=x.mean(),Q=Q,m=m,
            maj=(np.mean(x>=5/9)>0.5), mad=np.median(np.abs(x-med)), raw=r.Prerequisite_raw))
    return pd.DataFrame(rows)
