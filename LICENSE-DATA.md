# Data and derived files (package version 2.1.0)

This record contains no raw learner data and no derived edge list.

The derived edge lists (`edge_list/fpkg_edges_crowd.csv`, `edge_list/fpkg_edges_expert.csv`) would reproduce
exercise names and crowd ratings from the Junyi Academy release (Chang, Hsu & Chen, 2015; PSLC DataShop
dataset 1198). The release is made available for non-commercial research; the author has not established that
its terms permit redistribution of the ratings in a derived form. Both files are therefore rebuilt locally from
the release by `code/make_edge_list.py` (invoked by `run_all.sh`) and are not distributed.

Numeric outputs in `expected_outputs/` and `supplementary_outputs/` are the author's own computed results and
are released under CC BY 4.0. They contain a small number of exercise names and rating strings quoted from
the release for illustration (the seven relations of the worked example in `worked_example.json` and
`out_card.json`, three exercise names in `expert_repairs.txt`, two edges in `audit_cphnb.json`); those
quotations remain under the terms of the release.

Users must obtain the release from PSLC DataShop (https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198)
or the mirror named in README.md and must cite:

Chang, H.-S., Hsu, H.-J., & Chen, K.-T. (2015). Modeling exercise relationships in e-learning:
A unified approach. Proceedings of the 8th International Conference on Educational Data Mining, 532–535.
