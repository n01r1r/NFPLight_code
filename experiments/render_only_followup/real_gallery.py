"""Add approved real A–D panels beside the preserved three-way report."""
import html
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from experiments.regularization.config import atomic_write_json

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'artifacts/relmap_three_way_real_20261019'
SUITE=ROOT/'artifacts/render_weight_followup_20261019'
D_RUN=ROOT/'artifacts/render_only_followup_20261019/run'
ARMS=('A_weight1_lr1e4','B_weight1_lr1e5','C_weight10_lr1e5')
START='<!-- render-followup-comparison:start -->'
END='<!-- render-followup-comparison:end -->'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relative(path, root):
    return os.path.relpath(path,root).replace('\\','/')


def available(run):
    path=run/'real_report/report_checks.json'
    return path.is_file() and read(path).get('status')=='passed'


def image_cell(path, caption, assets, gamma=None):
    if not path.is_file():
        raise FileNotFoundError(path)
    assets.append(path)
    linear=html.escape(relative(path,TARGET),quote=True)
    attr=''
    if gamma is not None:
        if not gamma.is_file():
            raise FileNotFoundError(gamma)
        assets.append(gamma)
        attr=f' data-linear-src="{linear}" data-gamma-src="{html.escape(relative(gamma,TARGET),quote=True)}"'
    return f'<td><img width="192" height="192" loading="lazy" alt="{html.escape(caption,quote=True)}" src="{linear}"{attr}></td>'


def replace_section(text, section):
    if START in text:
        if text.count(START)!=1 or text.count(END)!=1:
            raise ValueError('Ambiguous followup section boundaries')
        a,b=text.index(START),text.index(END)+len(END)
        return text[:a]+section+text[b:]
    if '</h1>' not in text:
        raise ValueError('Report title boundary missing')
    return text.replace('</h1>','</h1>'+section,1)


def build():
    runs={arm:SUITE/'runs'/arm for arm in ARMS}
    runs['D_render_only_lr1e5']=D_RUN
    readiness={arm:available(run) for arm,run in runs.items()}
    marker=TARGET/'render_followup_checks.json'
    if marker.exists() and read(marker).get('schema_version')==2 and read(marker).get('ready')==readiness:
        return read(marker)
    backup=TARGET/'index_before_render_followup.html'
    if not backup.exists():
        shutil.copy2(TARGET/'index.html',backup)
    records=read(TARGET/'results.json')
    assert len(records)==5
    assets=[]
    columns=[('old64m','기존 relmap 64M',None),('old7m','기존 relmap 7M',None),
             ('new7m','변경 relmap 7M · render49k',None)]
    # The shared map baseline makes changes introduced by each followup visible.
    if readiness[ARMS[0]]:
        columns.append(('clean','Clean map135k',runs[ARMS[0]]))
    columns.extend((arm,arm.split('_')[0]+(' · render-only' if arm.startswith('D_') else
                    ' · λ10 / LR1e-5' if arm.startswith('C_') else
                    ' · λ1 / LR1e-5' if arm.startswith('B_') else ' · λ1 / LR1e-4'),run)
                   for arm,run in runs.items())
    body=START+'<section id="render-followup-comparison"><h2>기존 3조건 옆 A–C · clean baseline · render-only</h2>'
    body+='<p>같은 실측 5묶음·both·공통 정합/프레임·author_patch20·FP32 결과입니다. 가로로 스크롤해 같은 항목을 비교합니다. C/D 대기는 결과가 아닙니다. 완료 후 이 파일을 새로고침하면 채워집니다. RGB는 공통 [0,1], normal=(unit XYZ+1)/2입니다. S head는 재렌더의 고정 F0=.04와 구분합니다. 실측 GT가 없어 물성 정확도를 판정하지 않습니다.</p>'
    body+='<label><input id="followup-gamma" type="checkbox">추가 비교 RGB 표시 gamma2.2 (기본 off, 수치 불변)</label>'
    body+='<p>합성 validation으로 checkpoint를 선정합니다. 기존 3조건은 학습 이력·선정 시점이 달라 A–D와의 차이를 relmap 또는 loss 하나의 효과로 단정하지 않습니다. 조명별 ΔE00 heatmap은 원래 보고서의 범위가 달라 여기서는 수치만 비교합니다.</p>'
    body+='<ul>'
    for arm,run in runs.items():
        label=arm+': '+('실측 완료' if readiness[arm] else '학습/보고 완료 대기')
        body+='<li>'+html.escape(label)
        if readiness[arm]:
            body+=' · <a href="'+html.escape(relative(run/'real_report/index.html',TARGET),quote=True)+'">조건별 상세 보고</a>'
        body+='</li>'
    body+='</ul>'
    metrics=[]
    for original in records:
        group=original['group']
        body+='<h3>'+html.escape(group)+'</h3><div style="overflow-x:auto"><table><thead><tr><th>항목</th>'
        body+=''.join('<th style="min-width:192px">'+html.escape(label)+'</th>' for _,label,_ in columns)
        body+='</tr></thead><tbody>'
        data={}
        for key,label,run in columns:
            if run is None:
                data[key]={'folder':TARGET/group/key,'metrics':original['models'][key]['common'],'old':True}
            elif key=='clean' or readiness[key]:
                report=run/'real_report'
                comparison=read(report/group/'comparison.json')
                if comparison['selected_indices']!=original['selected_indices']:
                    raise ValueError('Different capture frame population: '+group+'/'+key)
                manifest=read(report/group/'manifest.json')
                gain=manifest['gain_measurement']
                # Historical summary divides Python floats; actual capture gain divides FP32 tensors.
                if abs(gain['gain']-original['gain'])>2e-7*max(1.,abs(original['gain'])):
                    raise ValueError('Different FP32 capture gain: '+group+'/'+key)
                for actual,expected in ((gain['near_patch_rgb_mean'],original['near_patch_mean']),
                        (gain['far_patch_rgb_mean'],original['far_patch_mean'])):
                    if actual!=expected:
                        raise ValueError('Different input/gain: '+group+'/'+key)
                old_manifest=read(TARGET/group/'new7m/manifest.json')
                if manifest['color_transforms']!=old_manifest['color_transforms']:
                    raise ValueError('Different capture color transforms')
                if read(report/group/'preparation/registration.json')!=read(TARGET/group/'preparation/registration.json'):
                    raise ValueError('Different capture registration')
                selection='best_map' if key=='clean' else 'best_render'
                weights=comparison['weights'][selection]
                data[key]={'folder':report/group/selection,'metrics':weights['reconstruction_difference'],
                           'old':False,'selection':weights['selection']}
        fields=[('Relmap','both/display/relation.png',False),('Log-relmap','both/display/relation_log.png',False),
                ('Normal','normal',False),('Diffuse','both/display/prediction_diffuse_rgb.png',True),
                ('Roughness','both/display/prediction_roughness.png',False),
                ('S head','both/display/prediction_specular_rgb.png',True),('Near 재렌더','near',True),('Far 재렌더','far',True)]
        for title,field,gamma in fields:
            body+='<tr><th>'+title+'</th>'
            for key,label,run in columns:
                if key not in data:
                    body+='<td>완료 대기</td>'
                    continue
                d=data[key]
                if field=='normal':
                    filename='unit_normal.png' if d['old'] else 'both/display/prediction_normal_rgb.png'
                elif field in ('near','far'):
                    filename='common_'+field+'.png' if d['old'] else 'render/'+field+'.png'
                else:
                    filename=field
                path=d['folder']/filename
                gamma_path=None
                if gamma:
                    gamma_path=d['folder']/(('common_'+field+'_gamma.png' if d['old'] else 'render_gamma22/'+field+'.png')
                        if field in ('near','far') else filename.replace('/display/','/display_gamma22/'))
                body+=image_cell(path,group+' '+label+' '+title,assets,gamma_path)
            body+='</tr>'
        for side in ('near','far'):
            body+='<tr><th>'+side+' 관측 L1 / ΔE00</th>'
            for key,label,run in columns:
                if key not in data:
                    body+='<td>완료 대기</td>'
                    continue
                d=data[key];m=d['metrics'][side]
                l1=m['l1'] if d['old'] else m['linear_rgb_l1']
                e00=m['delta_e00'] if d['old'] else m['delta_e00_mean']
                metrics.append({'group':group,'condition':key,'side':side,'observed_l1':l1,'observed_e00':e00})
                body+=f'<td>{l1:.7g} / {e00:.5f}</td>'
            body+='</tr>'
        body+='</tbody></table></div>'
    body+='</section><script>document.getElementById("followup-gamma").addEventListener("change",e=>{document.querySelectorAll("#render-followup-comparison img[data-gamma-src]").forEach(i=>i.src=e.target.checked?i.dataset.gammaSrc:i.dataset.linearSrc)});</script>'+END
    text=(TARGET/'index.html').read_text(encoding='utf-8')
    updated=replace_section(text,body)
    temporary=TARGET/'index.followup.tmp'
    temporary.write_text(updated,encoding='utf-8')
    os.replace(temporary,TARGET/'index.html')
    checks={'schema_version':2,'passed':True,'ready':readiness,'groups':5,'columns':len(columns),'image_references':len(assets),
            'shared_registration_frames_patch_means_color_transforms':True,'existing_three_way_results_preserved':True,
            'gain_summary_rounding':'historical Python float ratio vs FP32 tensor ratio, relative tolerance2e-7; patch RGB means exact',
            'backup':str(backup),'updated_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'browser_checked':False,'sha256_computed':False,'metrics':metrics}
    atomic_write_json(marker,checks)
    return checks


if __name__=='__main__': build()
