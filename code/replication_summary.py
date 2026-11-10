"""Collect the cross-dataset numbers reported in Table 8 of the manuscript. Output: out_summary.json"""
import json
J = json.load(open('out_benchmark.json'))
import sys  # usage: python replication_summary.py ASSIST_DIR EDNET_DIR XES_DIR XES_MINED_DIR (folders written by xds_prep.py / xds_eval.py)
D = {'ASSISTments': json.load(open(f'{sys.argv[1]}/out_xds.json')),
     'EdNet': json.load(open(f'{sys.argv[2]}/out_xds.json')),
     'XES3G5M': json.load(open(f'{sys.argv[3]}/out_xds.json')),
     'XES3G5M_mined': json.load(open(f'{sys.argv[4]}/out_xds.json'))}
S = {}
# Junyi
E = J['E1_E3_estimators_protocols']; E2 = J['E2_stronger_learner_models']; E2b = J['E2b_attention_kt']
acc = J['estimator_calibration_unobserved']
kts = {'DKT': {p: E2[p]['B1RK_auc'] for p in E2}}
for kt in ['SAKT', 'AKT']: kts[kt] = {p: E2b[p][f'B1RK_{kt}_auc'] for p in E2b}
S['Junyi'] = dict(
    cases=J['n_test'], unobs_s=E['snapshot']['share_unobserved_713'], unobs_q=E['fully_sequential']['share_unobserved_713'],
    brier={k: v['brier'] for k, v in acc.items()}, auc={k: v['auc'] for k, v in acc.items()},
    pooled_zero_G=E['snapshot']['ZERO|coupled|B5'], esd_zero_G=E['snapshot']['ZERO|esd|B5'],
    B1=E2['snapshot']['B1_auc'], B1RK={k: v['snapshot'] for k, v in kts.items()}, B1RK_q={k: v['fully_sequential'] for k, v in kts.items()})
best = max(kts, key=lambda k: kts[k]['snapshot']); S['Junyi']['best_kt'] = best
def over(p):
    if best == 'DKT':
        return dict(zero=E2[p]['ZERO|esd|B5|total_vs_B1RK'], kt=E2[p]['DKT|esd|B5|total_vs_B1RK'], GU_zero=E2[p]['ZERO|esd|B5-B9'],
                    flags=J['E4b_graph_over_B1RK_flags_per_1000'][f'{p}|top20%'])
    return dict(zero=E2b[p][f'{best}|ESD-ZERO|B5_total_vs_B1RK'], kt=E2b[p][f'{best}|ESD-{best}|B5_total_vs_B1RK'],
                GU_zero=E2b[p][f'{best}|ESD-ZERO|B5-B9'], flags=J['E4b_graph_over_B1RK_flags_per_1000'][f'{p}|{best}|top20%'])
S['Junyi']['over_best'] = {p: over(p) for p in ['snapshot', 'fully_sequential']}
for nm, o in D.items():
    U = o['estimator_use_protocol']; L = o['learner_models']
    S[nm] = dict(cases=o['n_test'], learners=o['learners_test'], unobs_s=U['snapshot']['share_unobserved'], unobs_q=U['fully_sequential']['share_unobserved'],
                 mass_s=U['snapshot']['score_mass_unobserved'], brier={k: v['brier'] for k, v in o['estimator_accuracy'].items()},
                 auc={k: v['auc'] for k, v in o['estimator_accuracy'].items()}, n_acc=o['estimator_accuracy']['ZERO']['n'],
                 pooled_zero_G=U['snapshot']['ZERO|pooled|G'], esd_zero_G=U['snapshot']['ZERO|esd|G'],
                 B1=L['snapshot']['B1_auc'], B1RK={k: L['snapshot'][f'B1RK_{k}_auc'] for k in ['DKT', 'SAKT', 'AKT']},
                 B1RK_q={k: L['fully_sequential'][f'B1RK_{k}_auc'] for k in ['DKT', 'SAKT', 'AKT']}, best_kt=L['snapshot']['best_kt'],
                 over_best={p: dict(zero=L[p]['ESD-ZERO|G_total_vs_B1RK'], kt=L[p][f"ESD-{L[p]['best_kt']}|G_total_vs_B1RK"],
                                    GU_zero=L[p]['ESD-ZERO|G-U'], flags=L[p][f"ESD-{L[p]['best_kt']}|G_flags20_vs_B1RK"],
                                    delong=L[p][f"ESD-{L[p]['best_kt']}|G_delong_vs_B1RK"]) for p in L},
                 best_vs_B1={p: L[p][f"B1RK_{L[p]['best_kt']}_vs_B1"] for p in L})
json.dump(S, open('out_summary.json', 'w'), indent=1); print(json.dumps(S, indent=1)[:6000])
