"""Reuse the approved static report and feature/Delta E00 evaluation helpers."""
import html
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from benchmark_fabric_leather import Bank, prepare, unit_prediction, fixed_f0, ndrs_loss
from experiments.full_dataset_suite.run import Model
from experiments.full_dataset_suite.reporting import write_offline_report
from experiments.regularization.config import atomic_write_json
from model.relation_geometry import spatial_far
from train_fabric import fixed_light_positions, top_view_positions, render_batch, render_input_images_batched

ROOT=Path(__file__).resolve().parents[2]
TEMPLATE=ROOT/'artifacts/full_dataset_static_test_20261012'


def load_helper(name, filename):
    spec=importlib.util.spec_from_file_location(name,TEMPLATE/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def export(base, run, snapshot, results, gallery, contract):
    write_offline_report(run)
    text=(run/'offline_report.html').read_text(encoding='utf-8')
    section='<section id="official-test"><h2>고정 test 정량 · '+html.escape(run.name)+'</h2><p>FP32 / AMP·TF32 off. Fabric5·Leather7, 각16crop×30조명(seed10000). 유효픽셀 → crop → 폴더 동일 가중치 평균. Test는 보고 전용이며 선정·scheduler에 사용하지 않았습니다. 이전에 관측한 test이므로 독립 최종 검증으로 주장하지 않습니다. Validation novel12와 test30의 모집단은 다릅니다.</p>'
    rows=[]
    for label,result in results.items():
        rows.append((label+' / 전체',result['mean']))
        rows.extend((label+' / '+category,values) for category,values in result['category'].items())
    section+=base.table(rows)
    for label,result in results.items():
        section+='<h3>'+label+' — 폴더별16crop</h3>'+base.table([(r['category']+'/'+r['material_id'],r) for r in result['materials']])
    section+='</section><section id="test-qualitative"><h2>고정 test 정성</h2><p>모든12폴더 crop0·조명0 고정. GT / map135k / validation 선정 best-render. Normal=(XYZ+1)/2, packed map=(x+1)/2. 공통 표시 범위[0,1], render clipping은 표시만 적용합니다. S head와 rendering 고정 F0=.04는 구분합니다. Total step=상속된 map135000+이번 render update이며 과거 run의 total step과 동일 학습 경로를 뜻하지 않습니다.</p>'
    for folder,panels in gallery.items():
        section+='<h3>'+html.escape(folder)+'</h3><div class="table-scroll"><table><thead><tr><th>조건</th>'+''.join('<th>'+x+'</th>' for x in ['Normal','Diffuse','Roughness','S head','Render light0'])+'</tr></thead><tbody>'
        for label,images in panels.items():
            section+='<tr><td>'+html.escape(label)+'</td>'+''.join('<td><img width="256" height="256" alt="'+html.escape(folder+' '+label+' '+str(i))+'" src="'+uri+'"></td>' for i,uri in enumerate(images))+'</tr>'
        section+='</tbody></table></div>'
    section+='</section><section><h2>고정 평가 기록</h2><pre>'+html.escape(json.dumps(contract,ensure_ascii=False,indent=2))+'</pre></section>'
    text=text.replace('</main>',section+'</main>')
    text=text.replace('10초 갱신 ·','정적 snapshot · 자동 갱신 없음 ·')
    assert 'fetch(' not in text and 'setInterval(' not in text
    assert len(re.findall('src="data:image/png;base64,',text))==180
    assert not re.search(r'(?:src|href)=["\'](?!data:|#)',text)
    (base.OUT/'index.html').write_text(text,encoding='utf-8')


def trace(ext, base, model, maps, valid, expected=None):
    obs=render_input_images_batched(model,maps)*valid
    features,_,_=model.build_features(obs,valid_mask=valid)
    if expected is not None:
        assert torch.equal(features,expected), 'GT feature differs from actual network input'
    near,far=obs.chunk(2,1)
    scaled=9*far
    corrected=spatial_far(scaled,model.time_co_map,model.spatial_attenuation_factor)
    clipped=corrected.clamp(0,1)
    denominator=model.coefficient.clamp(min=1e-5)
    q=(clipped-near).abs().mean(1,keepdim=True)/denominator
    decoded=(features+1)/2
    assert torch.allclose(decoded[:,18:19],(1-model.ClipToOne(q,valid))*valid,atol=2e-7,rtol=0)
    assert torch.allclose(decoded[:,19:20],(1-model.ClipToOne(q.clamp(min=1e-5).log(),valid))*valid,atol=2e-7,rtol=0)
    rgb={'I_N (near)':near,'I_F (far)':far,'9 I_F raw':scaled,
         'K_theta * (9 I_F * spatial_factor) raw':corrected,'clip(corrected far)':clipped,
         'signed near - corrected far [-1,1]':(near-clipped+1)/2,
         'relation feature':decoded[:,18:19],'log-relation feature':decoded[:,19:20],
         'non-saturated feature mask':decoded[:,20:21],'valid GT support':valid.float(),
         'log near':decoded[:,9:12],'log far':decoded[:,12:15],'log aligned far':decoded[:,15:18]}
    tiles={name:ext.rgb_tiles(ext.hwc(x).squeeze(-1) if x.shape[1]==1 else ext.hwc(x)) for name,x in rgb.items()}
    signed=ext.hwc(near-clipped)
    tiles['signed near - corrected far [-1,1]'].update(minimum=float(signed.min()),maximum=float(signed.max()),display_range=[-1,1],display_clipped_fraction=0)
    scalars={'spatial inverse-square factor':model.spatial_attenuation_factor,'K_theta':model.time_co_map,
        'coefficient_raw':model.coefficient_raw,'coefficient normalized':model.coefficient,
        'denominator max(c,1e-5)':denominator,'relation q raw':q,'log q raw':q.clamp(min=1e-5).log()}
    tiles.update({name:ext.scalar_tile(ext.hwc(x)[...,0],name) for name,x in scalars.items()})
    return tiles


@torch.inference_mode()
def build(run):
    run=Path(run)
    out=run/'report'
    out.mkdir(exist_ok=False)
    frozen=out/'snapshot'
    frozen.mkdir()
    config=json.loads((run/'execution_config.json').read_text(encoding='utf-8'))
    completion=json.loads((run/'complete.json').read_text(encoding='utf-8'))
    assert completion['status']=='screening_complete'
    for name in ('best_map.pt','best_render.pt','execution_config.json'):
        shutil.copy2(run/name,frozen/name)
    base=load_helper('build_report','build_report.py')
    base.OUT=out
    base.export_report=lambda snapshot,results,gallery,contract:export(base,run,snapshot,results,gallery,contract)
    base.save('training_snapshot.json',json.loads((run/'offline_snapshot.json').read_text(encoding='utf-8')))
    rows=[json.loads(x) for x in Path(config['manifest']).read_text(encoding='utf-8').splitlines() if x.strip()]
    groups={s:{x['group_id'] for x in rows if x['split']==s} for s in ['train','val','test']}
    assert not groups['test']&(groups['train']|groups['val'])
    bank=Bank(config['manifest'],'test')
    assert len(bank.records)==12 and sum(x['category']=='Fabric' for x in bank.records)==5
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=Model(config['policy'],config['scientific_settings']['backbone'])
    lights=fixed_light_positions(30,seed=10000,device=model.device)
    views=top_view_positions(30,device=model.device)
    contract={'snapshot_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'run':str(run),'condition':config['scientific_settings']['loss'],
        'initial_render_lr':config['policy']['render_lr'],'selection':completion,'checkpoint_selection':'validation only',
        'claim_scope':'single seed screen; previously inspected test, not independent final evidence',
        'precision':'FP32 AMP/TF32 off','light_positions':lights.cpu().tolist(),'view_positions':views.cpu().tolist(),
        'test_folders':12,'crops':list(range(16)),'qualitative_crop':0,'qualitative_light':0,'sha256_computed':False}
    results={}
    gallery={}
    for label,name in [('best-map','best_map.pt'),('best-render','best_render.pt')]:
        checkpoint=torch.load(frozen/name,map_location='cpu',weights_only=True)
        model.net_g.load_state_dict(checkpoint['net_g'],strict=True)
        del checkpoint
        model.net_g.eval()
        materials=[]
        for i,record in enumerate(bank.records):
            crops=[]
            for crop in range(16):
                safe,valid,features=prepare(model,bank.crop(i,crop)[None])
                pred=unit_prediction(model.net_g(features))
                _,parts=ndrs_loss(pred,safe,valid)
                reference=render_batch(model,safe,lights,views)
                prediction=render_batch(model,fixed_f0(pred),lights,views)
                rendering=(prediction-reference).abs().masked_select(valid[:,None].expand_as(prediction)).mean()
                angle=torch.rad2deg(torch.acos((pred[:,:3]*safe[:,:3]).sum(1,keepdim=True).clamp(-1,1))).masked_select(valid).mean()
                values=[float(x) for x in parts]+[float(rendering),float(angle),float(sum(parts[:3])/3),float(sum(parts))]
                assert np.isfinite(values).all()
                crops.append(dict(zip(base.KEYS,values)))
                if crop==0:
                    assert torch.equal(pred,unit_prediction(model.net_g(features)))
                    panel=gallery.setdefault(record['category']+' / '+record['material_id'],{})
                    panel.setdefault('GT',base.previews(safe,reference))
                    panel[label]=base.previews(pred,prediction)
                del safe,valid,features,pred,reference,prediction
            materials.append({**{k:record[k] for k in ['material_id','category','group_id']},**base.mean(crops),'crops':crops})
            bank.cache.clear()
            print(label,i+1,record['material_id'],flush=True)
        results[label]={'mean':base.mean(materials),'materials':materials,'category':{c:{'materials':sum(x['category']==c for x in materials),**base.mean([x for x in materials if x['category']==c])} for c in ['Fabric','Leather']}}
    base.save('evaluation_contract.json',contract)
    base.save('test_metrics.json',results)
    base.save('qualitative_panels.json',gallery)
    del model,bank
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    ext=load_helper('followup_feature_report','extend_features.py')
    ext.OUT=out
    ext.trace=lambda model,maps,valid,expected=None:trace(ext,base,model,maps,valid,expected)
    original_export=ext.export
    def export_extension(extra, extension_contract):
        extension_contract.update(generated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
            relation_version=config['scientific_settings']['rendering']['relation_version'],
            K_I_F='spatial_far(9*far, time_co_map, spatial_attenuation_factor), before clipping')
        base.save('feature_delta_contract.json',extension_contract)
        original_export(extra,extension_contract)
    ext.export=export_extension
    ext.main()
    # The historical helper's renderer/metrics are reused; correct its geometry prose for this lineage.
    path=out/'index.html'
    text=path.read_text(encoding='utf-8')
    text=text.replace('K=9Kθ.', 'K=9Kθ·spatial_factor (중앙 정규화 거리비율).')
    text=text.replace('model.time_co_map * (9 * far), before clipping','spatial_far(9*far, time_co_map, spatial_attenuation_factor), before clipping')
    text=text.replace('K I_F = K_theta (9 I_F)', 'K I_F = K_theta (9 I_F spatial_factor)')
    path.write_text(text,encoding='utf-8')
    assert len(gallery)==12 and 'fetch(' not in text and 'setInterval(' not in text
    assert 'feature-maps' in text and 'render-e00' in text and 'render-differences' in text
    atomic_write_json(out/'verification.json',{'passed':True,'test_folders':12,'crops_per_folder':16,'fixed_crop':0,'fixed_light':0,
        'gt_feature_exact':True,'finite':True,'self_contained':True,'inline_pngs':text.count('src="data:image/png;base64,'),
        'feature_delta_e00_extension':True,'sha256_computed':False,'visual_browser_checked':False})
    print('REPORT_COMPLETE',path,flush=True)
