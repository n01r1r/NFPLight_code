"""Joint render/NDRS decisions from the matched validation records only."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.regularization.config import atomic_write_json

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'artifacts/render_weight_followup_20261019'
OUT=ROOT/'artifacts/render_only_followup_20261019'
KEYS=('render_l1','map_normal','map_diffuse','map_roughness','map_specular','ndrs_sum','normal_degrees')
COMPONENTS=KEYS[1:5]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def differences(current, reference):
    result={}
    for key in KEYS:
        a,b=current[key],reference[key]
        if not np.isfinite([a,b]).all() or min(a,b)<0:
            raise ValueError('Invalid validation metric: '+key)
        result[key]={'absolute':a-b,'relative_percent':100*(a/b-1) if b else None}
    return result


def decide(baseline, selected, final):
    d=selected['D']
    render_better=all(d['render_l1']<selected[k]['render_l1'] for k in ('B','C')) and d['render_l1']<baseline['render_l1']
    component_no_worse=all(d[k]<=min(baseline[k],selected['B'][k],selected['C'][k]) for k in COMPONENTS)
    end_better=all(final['D']['render_l1']<final[k]['render_l1'] for k in ('B','C'))
    guard=d['ndrs_sum']<=1.02*baseline['ndrs_sum']
    if render_better and component_no_worse and end_better and guard:
        decision='promote_baseline_vs_render_only_screening'
        explanation='선택 validation render와 10k 끝점이 B/C보다 낮고 N/D/R/S 각각도 baseline/B/C보다 나쁘지 않아, 이 seed·예산의 주 비교를 clean baseline→render-only로 제안한다.'
    elif render_better and guard:
        decision='render_gain_with_tradeoff_keep_primary_pending'
        explanation='Render 이득은 있지만 성분별 악화 또는 10k 끝점 일관성 문제가 있어 주 비교 교체를 유보한다. 합×1.02 통과는 성분별 비열등을 보장하지 않는다.'
    else:
        decision='keep_existing_primary_render_only_not_superior_in_screen'
        explanation='이 통제 screening에서 render-only의 B/C 대비 공동 우위가 확인되지 않아 기존 주 비교를 유지한다. 다른 LR·장기 학습의 render-only 전체를 기각하지 않는다.'
    return {'decision':decision,'explanation':explanation,'render_selected_better_than_baseline_B_C':render_better,
            'all_four_components_no_worse_than_baseline_B_C':component_no_worse,
            'render_final10k_better_than_B_C':end_better,'ndrs_guard_1_02':guard,
            'scope':'single seed selected-validation screening, descriptive criterion; not significance or independent generalization proof'}


def paired_groups(current, reference):
    def group(rows):
        answer={}
        for row in rows:
            answer.setdefault(row['group_id'],[]).append(row)
        return {k:{m:float(np.mean([v[m] for v in values])) for m in KEYS} for k,values in answer.items()}
    a,b=group(current),group(reference)
    if a.keys()!=b.keys():
        raise ValueError('Different validation source groups')
    return {key:{'groups':len(a),'D_minus_reference_mean':float(np.mean([a[g][key]-b[g][key] for g in a])),
                 'D_lower_groups':sum(a[g][key]<b[g][key] for g in a),
                 'D_higher_groups':sum(a[g][key]>b[g][key] for g in a)} for key in KEYS}


def build():
    runs={'A':SUITE/'runs/A_weight1_lr1e4','B':SUITE/'runs/B_weight1_lr1e5',
          'C':SUITE/'runs/C_weight10_lr1e5','D':OUT/'run'}
    selected={}; final={}; histories={}; selected_entries={}
    for label,run in runs.items():
        completion=read(run/'complete.json')
        if completion['status']!='screening_complete' or completion['render_updates']!=10000:
            raise ValueError('Condition not fully complete: '+label)
        entries=read(run/'evaluation.json')
        history={r['step']:r for r in entries if r['stage']=='render'}
        if set(history)!=set(range(0,10001,1000)):
            raise ValueError('Different evaluation population/timing: '+label)
        histories[label]=history
        step=0 if completion['selected']=='best_map' else completion['selection_step']
        selected_entries[label]=history[step]
        selected[label]=history[step]['populations']['heldout']['mean']
        final[label]=history[10000]['populations']['heldout']['mean']
        if selected[label]['render_l1'] != completion['render_l1']:
            raise ValueError('Selection metrics inconsistent: '+label)
    baseline=histories['B'][0]['populations']['heldout']['mean']
    reference_population=histories['B'][0]['populations']['heldout']
    baseline_ids={(x['group_id'],x['material_id'],x['category']) for x in reference_population['materials']}
    for label,history in histories.items():
        if differences(history[0]['populations']['heldout']['mean'],baseline)!=differences(baseline,baseline):
            raise ValueError('Different initial validation: '+label)
        for entry in history.values():
            pop=entry['populations']['heldout']
            if pop['crop_indices']!=[0,4,8,12] or pop['novel_lights']!=12 or pop['sample_folders']!=164:
                raise ValueError('Different heldout contract')
            if {(x['group_id'],x['material_id'],x['category']) for x in pop['materials']}!=baseline_ids:
                raise ValueError('Different heldout material identities')
    result={'baseline':baseline,'selected':selected,'final10k':final,
            'selected_D_minus':{k:differences(selected['D'],v) for k,v in {'baseline':baseline,'B':selected['B'],'C':selected['C']}.items()},
            'paired_source_groups':{k:paired_groups(selected_entries['D']['populations']['heldout']['materials'],
                selected_entries[k]['populations']['heldout']['materials']) for k in ('B','C')},
            'decision':decide(baseline,selected,final),'independent_test':False,'real_data_used_for_selection':False,
            'matching_verified':read(OUT/'run/matching_verification.json'),'sha256_computed':False}
    atomic_write_json(OUT/'comparison.json',result)
    lines=['# Render-only · B/C 공동 비교','',result['decision']['explanation'],'',
           '단일 seed·동일10k screening이며 수렴·통계적 유의성·독립 최종 일반화 검증이 아니다. 선택 validation과10k끝점을 구분한다. 기존 test는 이미 열람했으며 보고용이다. 실측 GT가 없고 실측은 checkpoint·주 비교 선정에 사용하지 않는다.','',
           '| 조건 | render L1 | N | D | R | S | NDRS | 평균 normal 각도 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,values in {'clean map135k':baseline,**selected}.items():
        lines.append('| '+label+' | '+' | '.join(f'{values[k]:.9g}' for k in KEYS)+' |')
    lines+=['','## D − baseline/B/C: 음수는 해당 오류 감소','',
            '| 참조 | render 변화% | N 변화% | D 변화% | R 변화% | S 변화% | NDRS 변화% | 각도 변화% |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,values in result['selected_D_minus'].items():
        lines.append('| '+label+' | '+' | '.join('—' if values[k]['relative_percent'] is None else f"{values[k]['relative_percent']:+.4f}" for k in KEYS)+' |')
    lines+=['','S의 기준 오차가 매우 작으므로 상대%와 절대 변화(JSON)를 함께 읽는다. NDRS 합 보호는 각 성분 비열등 보장이 아니다.',
            '', 'A–C 원시 결과·가중치는 유지한다. 주 비교 변경은 이 screening의 보고상 판단이며 기본 inference 모델 교체가 아니다.',
            '', '[D 합성 정성](run/report/index.html) · [D 실측](run/real_report/index.html) · [공통 실측 gallery](../relmap_three_way_real_20261019/index.html)']
    (OUT/'COMPARISON_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'validation_trajectories.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','render_step',*KEYS]);writer.writeheader()
        for label,history in histories.items():
            for step,entry in history.items():
                writer.writerow({'arm':label,'render_step':step,**{k:entry['populations']['heldout']['mean'][k] for k in KEYS}})
    return result


if __name__=='__main__': build()
