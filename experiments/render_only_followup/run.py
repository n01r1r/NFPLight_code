"""Matched render-only branch; never modifies or restarts the A–C queue."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from itertools import zip_longest

import psutil
import torch
from benchmark_fabric_leather import prepare, unit_prediction, ndrs_loss
from experiments.full_dataset_suite.config import file_identity
from experiments.full_dataset_suite.engine import Engine, atomic_save, load_checkpoint, capture_rng
from experiments.full_dataset_suite.run import Pipeline, render_loss, verify_smoke_restore
from experiments.full_dataset_suite.storage import require_disk_reserve
from experiments.full_dataset_suite.reporting import write_offline_report, inference_payload
from experiments.regularization.config import atomic_write_json
from experiments.render_weight_followup.run import initialize_render
from experiments.robustness_suite.run import process_lock, memory_observation, memory_is_low, runtime_environment

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT/'artifacts/render_weight_followup_20261019'
OUT = ROOT/'artifacts/render_only_followup_20261019'
RUN = OUT/'run'
REFERENCES = ('B_weight1_lr1e5', 'C_weight10_lr1e5')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_sources(config):
    for item in config['file_identities']:
        if file_identity(item['path']) != item:
            raise ValueError('Source/input identity changed: '+item['path'])


def prepare_run():
    OUT.mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=True)
    config = copy.deepcopy(read(SUITE/'runs'/REFERENCES[0]/'execution_config.json'))
    validate_sources(config)
    config.update(package='render_only_followup', authority='2026-10-19 user: preserve A-C; after C, map135k render-only LR1e-5/10k; matched controls and joint render/NDRS comparison')
    config['policy']['arm'] = 'D_render_only_lr1e5'
    config['scientific_settings']['loss'].update(map_weight=0.0, render_weight=1.0,
        training_objective='render only; N/D/R/S diagnostics detached, no map backward',
        map_terms_role='evaluation and logging only; inherited map baseline and guard retained')
    config['followup'].update(comparison_arms=list(REFERENCES),
        start_after='A-C synthetic queue and C real report complete',
        test_use='report only; already inspected, not independent final evidence')
    config['file_identities'].extend(file_identity(p) for p in sorted(Path(__file__).parent.glob('*.py')))
    path = RUN/'execution_config.json'
    if path.exists():
        if read(path) != config:
            if (OUT/'queue.json').exists() or (RUN/'last.pt').exists() or (OUT/'smoke/last.pt').exists():
                raise ValueError('Started render-only contract changed; preserve existing artifact')
            atomic_write_json(OUT/'previous_prelaunch_config.json',read(path))
            atomic_write_json(path,config)
    else:
        atomic_write_json(path, config)
    atomic_write_json(RUN/'display_history.json', read(SUITE/'runs'/REFERENCES[0]/'display_history.json'))
    return config


def render_objective(rendered, maps):
    # Map value remains observable without contributing even a zero-weight graph.
    return rendered


class RenderOnlyPipeline(Pipeline):
    references = ()

    def objective(self, network, indices, transforms, stage):
        if stage != 'render':
            raise ValueError('Render-only branch cannot perform map updates')
        for name, stream in self.references:
            line = stream.readline()
            if not line:
                raise ValueError('Reference batch log ended: '+name)
            row = json.loads(line)
            if row['indices'] != indices or row['d4_transforms'] != transforms:
                raise ValueError('Batch/D4 mismatch before forward: '+name)
        target = torch.stack([self.train.crop(i//16, i%16, t%4, t>=4) for i,t in zip(indices,transforms)])
        safe, valid, features = prepare(self.model, target)
        pred = unit_prediction(network(features))
        maps, parts = ndrs_loss(pred.detach(), safe, valid)
        rendered = render_loss(self.model, pred, safe, valid, self.training_lights)
        return render_objective(rendered, maps), {
            'ndrs_sum':float(maps), 'map_loss':float(maps), 'map_weight':0.,
            'ndr_mean':float(sum(parts[:3])/3), 'render_loss':float(rendered.detach()),
            'weighted_render_loss':float(rendered.detach()), 'render_weight':1.,
            'map_aggregation':'ndrs_sum', 'objective_mode':'render_only',
            **dict(zip(('normal_loss','diffuse_loss','roughness_loss','specular_loss'),map(float,parts)))}


def verify_start(engine, best, end, config):
    assert not engine.optimizer.state
    assert engine.optimizer.param_groups[0]['lr'] == 1e-5
    assert engine.betas == (0.9,0.999) and engine.clip_norm == 1.
    verify_smoke_restore(best['net_g'], {k:v.detach().cpu() for k,v in engine.net.state_dict().items()}, 'initial_weights')
    verify_smoke_restore(end['rng'], capture_rng(), 'initial_global_rng')
    for key, actual in (('sample_rng',engine.samples.getstate()),('d4_rng',engine.d4.getstate()),
                        ('augmentation_rng',engine.augmentation_rng.getstate()),('noise_rng',engine.noise_generator.get_state())):
        verify_smoke_restore(end[key],actual,key)
    expected = read(SUITE/'runs'/REFERENCES[0]/'execution_config.json')
    a,b = copy.deepcopy(config['policy']),copy.deepcopy(expected['policy'])
    a.pop('arm'); b.pop('arm')
    assert a == b


def compare_logs(left, right, count=10000):
    pairs = 0
    with Path(left).open() as a, Path(right).open() as b:
        for x,y in zip_longest(a,b):
            if x is None or y is None:
                raise ValueError('Different training log lengths')
            x,y = json.loads(x),json.loads(y)
            if any(x[k] != y[k] for k in ('step','indices','d4_transforms')):
                raise ValueError('Different batch/D4 sequence')
            pairs += 1
    if pairs != count:
        raise ValueError(f'Expected {count} matched steps, found {pairs}')
    return pairs


def train(*, smoke=False):
    config = read(RUN/'execution_config.json')
    validate_sources(config)
    folder = OUT/'smoke' if smoke else RUN
    folder.mkdir(exist_ok=True)
    if smoke:
        config = copy.deepcopy(config)
        config['purpose'] = 'smoke'
        atomic_write_json(folder/'execution_config.json',config)
    require_disk_reserve(folder)
    with process_lock(folder,resume=False) as process, ExitStack() as stack:
        atomic_write_json(folder/'runtime.json',{'environment':runtime_environment(),'process':process})
        pipe = RenderOnlyPipeline(config,smoke=smoke)
        pipe.references = [(name,stack.enter_context((SUITE/'runs'/name/'train.jsonl').open())) for name in REFERENCES]
        engine = Engine(folder,pipe.model.net_g,config['policy'],config,pipe.objective,pipe.evaluate,sample_count=9056)
        engine.memory_guard = lambda: memory_is_low(memory_observation())
        best = load_checkpoint(SUITE/'starting_snapshot/best_map.pt')
        end = load_checkpoint(SUITE/'starting_snapshot/map_stage_end.pt')
        initialize_render(engine,best,end)
        verify_start(engine,best,end,config)
        atomic_write_json(folder/'start_verification.json',{'passed':True,'weights':'selected map135k',
            'rng':'map_stage_end155k; sample/D4/augmentation/noise/CPU/CUDA states exactly equal',
            'fresh_adam':True,'initial_lr':1e-5,'betas':[.9,.999],'clip_norm':1.,'map_weight':0.,
            'parameter_count':sum(p.numel() for p in engine.net.parameters()),'sha256_computed':False})
        del best,end
        result = engine.run({'map_cap':0,'render_cap':2 if smoke else 10000})
        expected = 2 if smoke else 10000
        if result['status'] != 'budget_paused' or result['render_updates'] != expected:
            raise RuntimeError('Render-only stopped; no automatic restart: '+result['status'])
        if smoke:
            before = engine.snapshot()
            engine.restore(load_checkpoint(folder/'last.pt'))
            verify_smoke_restore(before,engine.snapshot())
            atomic_write_json(folder/'verification.json',{'passed':True,'render_updates':2,
                'new_map_updates':0,'exact_restore':True,'production_training_started':False,'sha256_computed':False})
        else:
            matching = {}
            actual = engine.snapshot()
            for arm in REFERENCES:
                matching[arm] = compare_logs(folder/'train.jsonl',SUITE/'runs'/arm/'train.jsonl')
                reference = load_checkpoint(SUITE/'runs'/arm/'last.pt')
                for key in ('rng','sample_rng','d4_rng','augmentation_rng','noise_rng'):
                    verify_smoke_restore(reference[key],actual[key],arm+'.'+key)
                del reference
            del actual
            engine.state.update(status='screening_complete',selected='best_render' if engine.state['best_render'] else 'best_map')
            engine.state['stop_reasons']['render']='fixed_10000_update_screening_budget; not convergence'
            engine.commit()
            selected = engine.state[engine.state['selected']]
            atomic_save(folder/'net_g_selected_fp32.pth', inference_payload(selected, config))
            atomic_write_json(folder/'matching_verification.json',{'passed':True,'matched_steps':matching,
                'final_global_sample_d4_augmentation_noise_rng_exact':True,'sha256_computed':False})
            atomic_write_json(folder/'complete.json',{'status':'screening_complete','selected':engine.state['selected'],
                'selection_step':selected['step'],'render_updates':10000,'ndrs_sum':selected['ndrs_sum'],
                'render_l1':selected['render_l1'],'stop_reason':engine.state['stop_reasons']['render']})
        write_offline_report(folder)


def prerequisites(training, real):
    if training.get('status') not in ('running','complete'):
        raise RuntimeError('A-C queue stopped; no render-only launch')
    if real.get('status') not in ('running','complete'):
        raise RuntimeError('A-C real report queue stopped; preserve partial results')
    return training['status']=='complete' and real['status']=='complete'


def worker():
    from experiments.bottleneck_fc.config import process_identity_matches
    from experiments.render_only_followup.real_gallery import build as gallery
    with process_lock(OUT,resume=False) as process:
        state={'status':'running','process':process,'phase':'waiting_C','automatic_restart':False,'sha256_computed':False}
        def update(phase):
            state['phase']=phase
            atomic_write_json(OUT/'queue.json',state)
        try:
            update('waiting_C')
            while True:
                training,real=read(SUITE/'queue.json'),read(SUITE/'real_reports/queue.json')
                gallery()
                if prerequisites(training,real):
                    break
                for ledger in (training,real):
                    if ledger['status']=='running':
                        p=ledger['process']
                        if not process_identity_matches(p['pid'],p['process_identity']):
                            raise RuntimeError('Predecessor process identity no longer live')
                time.sleep(30)
            # Resource checks and GPU work begin only after both predecessor queues finish.
            require_disk_reserve(RUN)
            if psutil.virtual_memory().available < 4*2**30:
                raise RuntimeError('Insufficient available RAM; no automatic retry')
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            free_mib=int(gpu.stdout.strip().splitlines()[0])
            atomic_write_json(OUT/'resources_before_gpu.json',{'available_ram_gib':psutil.virtual_memory().available/2**30,
                'disk_free_gib':psutil.disk_usage(str(ROOT)).free/2**30,'gpu_free_mib':free_mib,'gpu_minimum_mib':20000})
            if free_mib<20000:
                raise RuntimeError('GPU reserve below 20000 MiB after C; no automatic retry')
            for mode in ('smoke','train','compare','report','real'):
                update(mode)
                with (OUT/(mode+'.log')).open('x',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.render_only_followup.run',mode],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                gallery()
            state['status']='complete'
            update('finished')
        except BaseException:
            state.update(status='failed',error=traceback.format_exc())
            update('stopped')
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','watch','smoke','train','compare','report','real','gallery'))
    args=parser.parse_args()
    if args.mode=='prepare': prepare_run()
    elif args.mode=='watch': worker()
    elif args.mode in ('smoke','train'): train(smoke=args.mode=='smoke')
    elif args.mode=='compare':
        from experiments.render_only_followup.comparison import build
        build()
    elif args.mode=='report':
        from experiments.render_weight_followup.report import build
        build(RUN)
    elif args.mode=='real':
        import importlib.util
        path=ROOT/'artifacts/spatial_7m_real_20261019/run_report.py'
        spec=importlib.util.spec_from_file_location('render_only_real',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        output=RUN/'real_report';output.mkdir(exist_ok=False)
        module.main(source=RUN,output=output,label='D render-only LR1e-5',screening=True)
    else:
        from experiments.render_only_followup.real_gallery import build
        build()


if __name__=='__main__': main()
