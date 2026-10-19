import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def verify_objective(old, output):
    root = ROOT
    import sys,json
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from verify_compatibility import equal
    from nfplight.pipeline import TrainingPipeline
    from nfplight.checkpoint import read_config,decode
    from nfplight.data import map_loss
    from nfplight.storage import write_json
    from model.nfplight_model import NFPLightModel
    from experiments.full_dataset_suite.run import render_loss
    from benchmark_fabric_leather import prepare,ndrs_loss,unit_prediction
    class Reference(NFPLightModel):
        def init_network(self): pass
    torch.set_num_threads(2);torch.manual_seed(19)
    execution=json.loads((old/'artifacts/C_convergence_followup_20261019/run/execution_config.json').read_text())
    runtime=json.loads((root/'configs/runtime.json').read_text())
    pipeline=TrainingPipeline(execution,read_config(root/'configs/7m_c.json'),runtime,device='cpu')
    reference=Reference(SimpleNamespace(device='cpu',feature_version='raw_calibrated_gain9_v3',capture_precision=True))
    target=pipeline.train.crop(0,0)[None]
    safe,mask,features=prepare(reference,target)
    a,b,c=pipeline.prepare(target)
    equal((safe,mask,features),(a,b,c),'pipeline preparation')
    raw=torch.rand(1,10,256,256)*2-1
    results=[]
    for legacy in (True,False):
        value=raw.clone().requires_grad_()
        pred=unit_prediction(value) if legacy else decode(value,'unit_xyz')
        maps,parts=(ndrs_loss if legacy else map_loss)(pred,safe,mask)
        rendered=render_loss(reference,pred,safe,mask,pipeline.training_lights) if legacy else pipeline.render_loss(pred,safe,mask,pipeline.training_lights,execution['scientific_settings']['rendering']['training_lights'])
        loss=maps+10*rendered
        loss.backward()
        results.append((loss.detach(),maps.detach(),rendered.detach(),value.grad))
    equal(results[0],results[1],'full resolution 40-light C objective and gradient')
    write_json(output/'objective_final.json',{'passed':True,'device':'cpu','shape':[1,10,256,256],'lights':40,'render_weight':10,'preparation_and_loss_and_gradient_exact':True,'research_training_started':False})
    print('PASS full 256px C objective, all 40 lights and prediction gradients exact',flush=True)


def verify_phases(old, output):
    root = ROOT
    import sys,json,copy,tempfile
    from pathlib import Path
    import torch
    from nfplight.training import Trainer,capture_rng,restore_rng
    from nfplight.storage import write_json
    from verify_compatibility import equal
    from experiments.full_dataset_suite.engine import Engine
    policy=json.loads((root/'configs/training.json').read_text())['policy']
    policy.update(evaluation_interval=1,checkpoint_interval=1,floor_observations=2,map_lr=1e-5,render_lr=1e-6)
    runtime=json.loads((root/'configs/runtime.json').read_text());runtime.update(sample_count=8,batch_size=2,disk_reserve_bytes=0)
    identity={'policy':policy}
    torch.set_num_threads(2);torch.manual_seed(19)
    network=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.1),torch.nn.Linear(3,1))
    def objective(network,indices,transforms,stage):
        return (network(torch.tensor(list(zip(indices,transforms)),dtype=torch.float32)/8)-.3).square().mean(),{}
    def evaluate(network,stage,step):
        return {'heldout':{'mean':{'ndrs_sum':1.,'render_l1':.1}}}
    new=Trainer(copy.deepcopy(network),identity,objective,evaluate,runtime)
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        legacy=Engine(folder/'old',copy.deepcopy(network),policy,identity,objective,evaluate,sample_count=8,batch_size=2)
        rng=capture_rng()
        new.run(folder/'new',{'map_cap':10,'render_cap':10})
        expected=new.snapshot()
        restore_rng(rng)
        legacy.run({'map_cap':10,'render_cap':10})
        equal(expected,legacy.snapshot(),'complete map/render phase transition and stopping')
        equal(torch.load(folder/'new/map_stage_end.pt',weights_only=True),torch.load(folder/'old/map_stage_end.pt',weights_only=True),'map stage end')
    write_json(output/'phases_final.json',{'passed':True,'map_transition_render_stop_and_full_state_exact':True,'map_updates':expected['state']['map_updates'],'render_updates':expected['state']['render_updates'],'device':'cpu','fixture':'small linear/dropout model'})
    print('PASS phase transition, selected weights, stopping and full state exact')


def verify_resume(old, output):
    root = ROOT
    import sys, json, copy
    from pathlib import Path
    import torch
    from nfplight.cli import validate_resume,runtime_environment
    from nfplight.checkpoint import load_payload,read_config
    from nfplight.storage import identity,write_json
    from nfplight.training import Trainer
    path=old/'artifacts/C_convergence_followup_20261019/run/last.pt'
    before=identity(path)
    payload=load_payload(path)
    if identity(path)!=before: raise ValueError('checkpoint changed during read; retry')
    profile=read_config(root/'configs/7m_c.json')
    runtime=json.loads((root/'configs/runtime.json').read_text())
    validate_resume(payload,payload['identity'],profile,runtime,device='cuda')
    environment=runtime_environment('cuda')
    recorded=json.loads((path.parent/'runtime.json').read_text())['environment']
    if recorded!=environment: raise ValueError((recorded,environment))
    from nfplight.network import NFPLight
    from verify_compatibility import equal
    trainer = Trainer(NFPLight(**profile['network']), payload['identity'], None, None, runtime)
    trainer.restore(payload)
    restored = trainer.snapshot()
    for key in restored:
        equal(restored[key], payload[key], f'current C restored {key}')
    write_json(output/'resume_final.json',{'passed':True,'checkpoint':before,'step':payload['state']['total_step'],'runtime_and_all_data_identities_valid':True,'environment_exact':True,'full_checkpoint_restore_exact':True,'sha256_computed':False})
    print('PASS current C full resume gate and actual CUDA environment',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Legacy objective, phase and current C resume contract checks')
    parser.add_argument('--legacy-root', required = True)
    parser.add_argument('--out', required = True)
    args = parser.parse_args()
    old, output = Path(args.legacy_root).resolve(), Path(args.out).resolve()
    if output.exists():
        raise FileExistsError('verification requires a new report directory')
    output.mkdir(parents = True)
    sys.path[:0] = [str(ROOT), str(old)]
    verify_objective(old, output)
    verify_phases(old, output)
    verify_resume(old, output)
