"""Single-seed intermediate-capacity screen using the existing D4 trainer."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from experiments.regularization import run as trainer
from experiments.regularization.config import atomic_write_json, file_identity
from experiments.bottleneck_fc.pipeline import Bank, cache_slots, make_lights, ndr_loss, render_error, unit_prediction
from experiments.bottleneck_fc.run import evaluate
PARENT=ROOT/'artifacts/regularization_probe_20261009_checked'
DEFAULT=ROOT/'artifacts/intermediate_capacity_15m_seed19_20261009'
SCRIPT=Path(__file__).resolve()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def build_plan(out):
    parent=read(PARENT/'execution_config.json')
    for identity in parent['file_identities']:
        if file_identity(identity['path'])!=identity:
            raise ValueError('parent source/data identity changed: '+identity['path'])
    settings=copy.deepcopy(trainer.effective_settings(parent))
    settings['model']['backbone']['middle_blocks']=12
    settings['model']['parameters']=15085034
    settings['training']['seeds']=[19]
    settings['experiment']['training']['seeds']=[19]
    settings['experiment']['scientific_settings']['backbone']['middle_blocks']=12
    settings['experiment']['scientific_settings']['baseline_parameters']=15085034
    settings['experiment']['design']['dropout_probability']=0.0
    with torch.device('meta'):
        net=trainer.create_network(0.0,settings['model']['backbone'])
    assert trainer.parameter_count(net)==15085034
    control=PARENT/'seed19_d4_only'/'complete.json'
    assert read(control)['seed']==19 and read(control)['condition']=='d4_only'
    return dict(version=1,output=str(out.resolve()),seed=19,condition=dict(id='d4_only',label='15.09M + D4; seed19',d4=True,dropout=False),settings=settings,resolved_inputs=parent['resolved_inputs'],parent_execution=parent,control_identity=file_identity(control),source_identity=file_identity(SCRIPT),authorization='User approved intermediate model recommendation with a single seed; exclude seed31',claim='Single-seed capacity screen against 7.61M D4-only seed19; same fixed materials/crops; not a capacity law or generalization proof')

def prepare(out):
    candidate=build_plan(out)
    path=out/'plan.json'
    if path.exists():
        if read(path)!=candidate:
            raise ValueError('immutable intermediate run identity mismatch')
    else:
        out.mkdir(parents=True,exist_ok=True)
        atomic_write_json(path,candidate)
        atomic_write_json(out/'status.json',dict(status='prepared',seed=19,parameters=15085034,scheduled_training_runs=1,training_started=False))
    return candidate

def verify_plan(plan):
    if file_identity(SCRIPT)!=plan['source_identity']:
        raise ValueError('scale-up runner changed after preparation')
    for identity in plan['parent_execution']['file_identities']+[plan['control_identity']]:
        if file_identity(identity['path'])!=identity:
            raise ValueError('source/data/control changed: '+identity['path'])

def wait_parent(out):
    atomic_write_json(out/'transition_status.json',dict(status='waiting_for_seed19_conclusion',pid=os.getpid(),parent=str(PARENT),skip_seed31=True))
    while not (PARENT/'two_seed_conclusion.json').exists():
        guard=read(PARENT/'seed19_guard_status.json')
        if guard['status']=='failed':
            raise RuntimeError('seed19 guard failed: '+guard.get('error','unknown'))
        time.sleep(2)
    scope=read(PARENT/'user_scope_completion.json')
    if not scope['user_scope_complete'] or scope['seed31_train_rows']!=0:
        raise RuntimeError('seed19-only scope not verified or seed31 was trained')
    from experiments.bottleneck_fc.config import process_identity_matches
    status=read(PARENT/'launch.json')
    parent_runtime=read(PARENT/'runtime.json')['invocations'][-1]
    if process_identity_matches(status['pid'],parent_runtime['process_identity']):
        raise RuntimeError('previous trainer is still alive')
    atomic_write_json(out/'transition_status.json',dict(status='parent_concluded_starting_scaleup',pid=os.getpid(),conclusion=str(PARENT/'two_seed_conclusion.json')))

def smoke(model,settings,condition,train_population,raw_train,lights):
    import random
    indices=trainer.sample_order(19,settings)[0]
    rng=random.Random(19+settings['experiment']['training']['d4_seed_offset'])
    safe,valid,features,_=trainer._condition_batch(model,condition,indices,train_population,raw_train,rng,settings)
    optimizer=trainer._optimizer(model,settings,'map')
    values=[]
    for stage in ('map','render'):
        optimizer.zero_grad(set_to_none=True)
        prediction=unit_prediction(model.net_g(features))
        map_loss,_=ndr_loss(prediction,safe,valid)
        render_loss=render_error(model,prediction,safe,valid,lights,settings['rendering']) if stage=='render' else map_loss.new_zeros(())
        loss=map_loss+render_loss
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.net_g.parameters(),1.0,error_if_nonfinite=True)
        assert torch.isfinite(loss) and torch.isfinite(grad) and grad>0
        optimizer.step()
        values.append(dict(stage=stage,loss=float(loss.detach()),gradient_norm=float(grad),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return values

def run(out,plan,resume=False,wait=False):
    verify_plan(plan)
    if wait:
        wait_parent(out)
    elif not (PARENT/'two_seed_conclusion.json').exists():
        raise RuntimeError('seed19 conclusion must precede scale-up')
    state=read(out/'status.json')
    if state['status']=='complete':
        return
    from experiments.bottleneck_fc.config import process_identity, process_identity_matches
    pid, identity = state.get('pid'), state.get('process_identity')
    if process_identity_matches(pid, identity) or (identity is None and process_identity(pid) is not None):
        raise RuntimeError('run process is active or unverified; do not resume twice')
    if state['status'] not in ('prepared','failed','interrupted') and not resume:
        raise RuntimeError('active/partial run; do not start twice')
    from train_fabric import require_scheduler_ready
    require_scheduler_ready(historical_resume=resume)
    settings=plan['settings']; condition=plan['condition']
    trainer._set_deterministic_settings(settings)
    environment=trainer._runtime_environment(settings)
    invocation=trainer._record_invocation(out,environment)
    verify_plan(plan)
    bank=Bank(plan['resolved_inputs']['manifest'],settings['dataset'])
    def setup():
        trainer._seed(19)
        model=trainer.RegularizationModel(settings,0.0,settings['runtime']['device'])
        assert trainer.parameter_count(model.net_g)==15085034
        populations={label:cache_slots(model,bank,plan['resolved_inputs'][label],.04) for label in ('train','unseen_crop','heldout')}
        raw=[bank.crop(slot) for slot in plan['resolved_inputs']['train']]
        lights=make_lights(settings['rendering']['training_lights'],model.device)
        novel=make_lights(settings['rendering']['novel_lights'],model.device)
        return model,populations,raw,lights,novel
    if not resume:
        model,populations,raw,lights,novel=setup()
        try:
            results=smoke(model,settings,condition,populations['train'],raw,lights)
            atomic_write_json(out/'gpu_smoke.json',dict(status='passed',effective_batch=4,fp32=True,steps=results,training_model_reinitialized_after_smoke=True))
        finally:
            del model,populations,raw,lights,novel
            torch.cuda.empty_cache()
    model,populations,raw,lights,novel=setup()
    # Reuse the proven stage loop, but report this one-run scope accurately.
    original=trainer._write_run_status
    def scoped_status(*args,**kwargs):
        row=original(*args,**kwargs)
        root=read(out/'status.json')
        scope=dict(scheduled_training_runs=1,comparison_expected_runs=2,selected_condition_ids=['d4_only'],selected_seeds=[19],parameters=15085034,run_scope='single-seed intermediate capacity screen; 7.61M D4 seed19 reference')
        root.update(scope)
        row.update(scope)
        atomic_write_json(out / f"seed{row['seed']}_{row['condition']}" / 'status.json',row)
        atomic_write_json(out/'status.json',root)
        return row
    trainer._write_run_status=scoped_status
    try:
        result=trainer._run_condition(output_dir=out,execution=plan['parent_execution'],settings=settings,invocation=invocation,seed=19,condition=condition,train_population=populations['train'],raw_train=raw,populations=populations,train_lights=lights,novel_lights=novel,model=model,resume=resume,identity=plan)
        # Save actual final FP32 maps using the existing evaluator/export contract.
        evaluate(model,populations,novel,settings,save_dir=out/'seed19_d4_only'/'arrays')
        atomic_write_json(out/'complete.json',result)
        control=read(plan['control_identity']['path'])
        atomic_write_json(out/'comparison.json',dict(seed=19,control_parameters=7609834,new_parameters=15085034,control=control['final'],scaleup=result['final'],actual_updates=dict(control_map=control['map_updates'],control_render=control['render_updates'],scaleup_map=result['map_updates'],scaleup_render=result['render_updates']),claim=plan['claim']))
    finally:
        trainer._write_run_status=original

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','wait-run','status'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=args.out.resolve()
    if args.command=='status':
        print(json.dumps(read(out/'status.json'),indent=2)); return
    if args.command in ("run", "wait-run"):
        from train_fabric import require_scheduler_ready
        require_scheduler_ready(historical_resume=args.resume)
    plan=prepare(out)
    if args.command=='prepare':
        print(json.dumps(dict(status='prepared',parameters=plan['settings']['model']['parameters'],seed=19,output=str(out)),indent=2)); return
    try:
        run(out,plan,args.resume,args.command=='wait-run')
    except BaseException as exc:
        from experiments.bottleneck_fc.config import process_identity, process_identity_matches
        state=read(out/'status.json')
        pid, identity = state.get('pid'), state.get('process_identity')
        if pid != os.getpid() and (process_identity_matches(pid, identity) or (identity is None and process_identity(pid) is not None)):
            raise
        atomic_write_json(out/'failure.json',dict(error_type=type(exc).__name__,error=str(exc)))
        state=read(out/'status.json'); state.update(status='failed',error=str(exc)); atomic_write_json(out/'status.json',state)
        raise
if __name__=='__main__':
    main()
