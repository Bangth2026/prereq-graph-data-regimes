import pandas as pd, networkx as nx
cr = pd.read_csv('../edge_list/fpkg_edges_crowd.csv'); ex = pd.read_csv('../edge_list/fpkg_edges_expert.csv')
def audit(name, edges):
    G = nx.DiGraph(); G.add_edges_from(edges)
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    inside = sum(1 for u, v in G.edges if any(u in c and v in c for c in sccs))
    two = sum(1 for u, v in G.edges if G.has_edge(v, u)) // 2
    print(name, G.number_of_nodes(), G.number_of_edges(), len(sccs), sorted((len(c) for c in sccs), reverse=True)[:10], inside, two, nx.is_directed_acyclic_graph(G))
E = list(zip(ex.source, ex.target)); acc = cr[cr.status == 'accepted']
audit('expert', E); audit('crowd_all', list(zip(cr.source, cr.target))); audit('crowd_admitted', list(zip(acc.source, acc.target)))
audit('merged', E + list(zip(acc.source, acc.target)))
Es = set(E); audit('merged_expert_direction_kept', E + [(s, t) for s, t in zip(acc.source, acc.target) if (t, s) not in Es])
