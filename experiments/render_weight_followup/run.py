"""Sequential render screening; preserve historical trainer and run artifacts."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import torch
from benchmark_fabric_leather import prepare as prepare_features, unit_prediction, ndrs_loss
from experiments.full_dataset_suite.config import execution_identity, file_identity
from experiments.full_dataset_suite.engine import Engine, atomic_save, load_checkpoint, restore_rng
from experiments.full_dataset_suite.run import Pipeline, render_loss, verify_smoke_restore
from experiments.full_dataset_suite.reporting import write_offline_report, inference_payload
from experiments.full_dataset_suite.storage import require_disk_reserve
from experiments.regularization.config import atomic_write_json
from experiments.robustness_suite.run import process_lock, runtime_environment, memory_observation, memory_is_low

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/render_weight_followup_20261019'
SOURCE = ROOT / 'artifacts/spatial_7m_retraining_20261017/runs/baseline7m_p01'
ARMS = {'A_weight1_lr1e4': (1., 1e-4), 'B_weight1_lr1e5': (1., 1e-5), 'C_weight10_lr1e5': (10., 1e-5)}


def combine_loss(maps, rendered, weight):
    if not 0 < weight < float('inf'):
        raise ValueError('render weight must be finite and positive')
    return maps + weight * rendered


class WeightedPipeline(Pipeline):
    def objective(self, network, indices, transforms, stage):
        if stage != 'render':
            raise ValueError('this experiment permits render updates only')
        target = torch.stack([self.train.crop(i//16, i%16, t%4, t>=4) for i,t in zip(indices, transforms)])
        safe, valid, features = prepare_features(self.model, target)
        pred = unit_prediction(network(features))
        maps, parts = ndrs_loss(pred, safe, valid)
        rendered = render_loss(self.model, pred, safe, valid, self.training_lights)
        weight = self.execution['scientific_settings']['loss']['render_weight']
        total = combine_loss(maps, rendered, weight)
        return total, {'ndrs_sum': float(maps.detach()), 'map_loss': float(maps.detach()),
            'ndr_mean': float(sum(parts[:3]).detach()/3), 'render_loss': float(rendered.detach()),
            'weighted_render_loss': float((weight*rendered).detach()), 'render_weight': weight,
            **dict(zip(('normal_loss','diffuse_loss','roughness_loss','specular_loss'), [float(x.detach()) for x in parts])),
            'map_aggregation': 'ndrs_sum'}


def initialize_render(engine, best, end):
    if best['stage'] != 'map' or best['step'] != 135000:
        raise ValueError('expected selected map135000')
    engine.net.load_state_dict(best['net_g'], strict=True)
    engine.state.update(stage='render', stage_step=0, total_step=135000, map_updates=135000,
                        render_updates=0, best_map=copy.deepcopy({k:v for k,v in best.items() if k not in ('identity','package','checkpoint_version','relation_version')}))
    engine.state['ancestral_map_updates'] = 135000
    engine.state['new_map_updates'] = 0
    engine.state['render_start_sample_offset'] = 155000
    engine._new_optimizer('render')
    engine.samples.setstate(end['sample_rng'])
    engine.d4.setstate(end['d4_rng'])
    engine.augmentation_rng.setstate(end['augmentation_rng'])
    engine.noise_generator.set_state(end['noise_rng'])
    restore_rng(end['rng'])


def prepare_suite():
    OUT.mkdir(parents=True, exist_ok=True)
    require_disk_reserve(OUT)
    frozen = OUT/'starting_snapshot'
    frozen.mkdir(exist_ok=True)
    identities = {}
    for name in ('best_map.pt','map_stage_end.pt','execution_config.json'):
        source = SOURCE/name
        identities[name] = file_identity(source)
        target = frozen/name
        if not target.exists():
            shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size or target.stat().st_mtime_ns != source.stat().st_mtime_ns:
            raise ValueError('starting snapshot identity mismatch')
    for arm, (weight, lr) in ARMS.items():
        folder = OUT/'runs'/arm
        folder.mkdir(parents=True, exist_ok=True)
        config = execution_identity('baseline7m_p01')
        config.update(package='render_weight_followup', authority='2026-10-19 user authorized sequential A/B/C and per-setting qualitative reports')
        config['policy'].update(arm=arm, render_lr=lr, render_cap=10000)
        config['scientific_settings']['loss']['render_weight'] = weight
        config['followup'] = {'initial_map_step':135000, 'new_map_updates':0, 'render_screening_updates':10000,
            'total_step_semantics':'135000 ancestral map updates + new render updates',
            'rng_source':'map_stage_end155000; identical for all arms; fresh render Adam',
            'starting_files':identities, 'test_use':'report only; not selection; previously inspected test is no longer independent final evidence'}
        config['file_identities'].extend(file_identity(p) for p in sorted(Path(__file__).parent.glob('*.py')))
        path = folder/'execution_config.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config and (folder/'last.pt').exists():
            raise ValueError('immutable followup config changed')
        atomic_write_json(path, config)
    return frozen


def train(arm, *, smoke=False):
    config = json.loads((OUT/'runs'/arm/'execution_config.json').read_text(encoding='utf-8'))
    for identity in config['file_identities']:
        if file_identity(identity['path']) != identity:
            raise ValueError('source/input identity changed since preparation: '+identity['path'])
    folder = OUT/('smoke' if smoke else 'runs')/arm
    folder.mkdir(parents=True, exist_ok=True)
    if smoke:
        config = copy.deepcopy(config)
        config['purpose'] = 'smoke'
        atomic_write_json(folder/'execution_config.json', config)
    require_disk_reserve(folder)
    with process_lock(folder, resume=False) as process:
        atomic_write_json(folder/'runtime.json', {'environment':runtime_environment(), 'process':process})
        pipe = WeightedPipeline(config, smoke=smoke)
        engine = Engine(folder, pipe.model.net_g, config['policy'], config, pipe.objective, pipe.evaluate, sample_count=9056)
        engine.memory_guard = lambda: memory_is_low(memory_observation())
        initialize_render(engine, load_checkpoint(OUT/'starting_snapshot/best_map.pt'), load_checkpoint(OUT/'starting_snapshot/map_stage_end.pt'))
        result = engine.run({'map_cap':0, 'render_cap':2 if smoke else 10000})
        if result['status'] != 'budget_paused' or result['render_updates'] != (2 if smoke else 10000):
            raise RuntimeError(f'screening did not reach authorized endpoint: {result["status"]}')
        if smoke:
            before = engine.snapshot()
            engine.restore(load_checkpoint(folder/'last.pt'))
            verify_smoke_restore(before, engine.snapshot())
            atomic_write_json(folder/'verification.json', {'passed':True,'render_updates':2,'new_map_updates':0,'exact_restore':True,'weight':10,'production_training_started':False})
        else:
            engine.state.update(status='screening_complete', selected='best_render' if engine.state['best_render'] else 'best_map')
            engine.state['stop_reasons']['render'] = 'fixed_10000_update_screening_budget; not convergence'
            engine.commit()
            selected = engine.state[engine.state['selected']]
            atomic_write_json(folder/'complete.json', {'status':'screening_complete', 'selected':engine.state['selected'],
                'selection_step':selected['step'], 'render_updates':10000, 'ndrs_sum':selected['ndrs_sum'],
                'render_l1':selected['render_l1'], 'stop_reason':engine.state['stop_reasons']['render']})
            atomic_save(folder/'net_g_selected_fp32.pth', inference_payload(selected, config))
        write_offline_report(folder)


def queue():
    with process_lock(OUT, resume=False) as process:
        ledger = {'status':'running','process':process,'arms':{},'sha256_computed':False}
        atomic_write_json(OUT/'queue.json', ledger)
        try:
            for arm in ARMS:
                ledger.update(current_arm=arm, phase='training')
                atomic_write_json(OUT/'queue.json',ledger)
                for mode in ('train','report'):
                    ledger['phase'] = mode
                    atomic_write_json(OUT/'queue.json',ledger)
                    with (OUT/f'{arm}_{mode}.log').open('x',encoding='utf-8') as log:
                        subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.render_weight_followup.run',mode,'--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                    if mode=='train' and arm != next(iter(ARMS)):
                        with (OUT/'runs'/next(iter(ARMS))/'train.jsonl').open() as a, (OUT/'runs'/arm/'train.jsonl').open() as b:
                            pairs=list(zip(a,b))
                        assert len(pairs)==10000
                        for left,right in pairs:
                            x,y=json.loads(left),json.loads(right)
                            assert x['indices']==y['indices'] and x['d4_transforms']==y['d4_transforms'] and x['step']==y['step']
                ledger['arms'][arm] = {'status':'complete','report':str(OUT/'runs'/arm/'report/index.html')}
                atomic_write_json(OUT/'queue.json',ledger)
            ledger.update(status='complete',phase='finished')
            atomic_write_json(OUT/'queue.json',ledger)
            comparison={}
            lines=['# Render weight/LR 10k 선별 비교','각 조건은 고정135k best-map에서 시작했다. Test는 보고 전용이며 독립 최종 검증이 아니다.','', '| 조건 | 선택 render step | validation NDRS | validation render L1 | 보고서 |','|---|---:|---:|---:|---|']
            for arm in ARMS:
                result=json.loads((OUT/'runs'/arm/'complete.json').read_text(encoding='utf-8'))
                comparison[arm]=result
                lines.append(f"| {arm} | {result['selection_step']} | {result['ndrs_sum']:.9f} | {result['render_l1']:.9f} | [HTML](runs/{arm}/report/index.html) |")
            atomic_write_json(OUT/'comparison.json',comparison)
            (OUT/'COMPARISON_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        except BaseException:
            ledger.update(status='stopped',error=traceback.format_exc(),automatic_restart=False)
            atomic_write_json(OUT/'queue.json',ledger)
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','smoke','train','report','queue'])
    parser.add_argument('--arm',choices=ARMS,default='C_weight10_lr1e5')
    args=parser.parse_args()
    if args.mode=='prepare': prepare_suite()
    elif args.mode=='queue': queue()
    elif args.mode=='report':
        from experiments.render_weight_followup.report import build
        build(OUT/'runs'/args.arm)
    else: train(args.arm,smoke=args.mode=='smoke')


if __name__=='__main__': main()
