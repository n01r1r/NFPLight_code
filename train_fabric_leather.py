"""FP32 Fabric+Leather training under the approved fixed-F0 contract.

Uses validated bank/step code from the speed probe; never reads official test
payloads, loads author/pretrained weights, or computes file hashes.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from model.nfplight_model import INPUT_NEAR, INPUT_FAR, INPUT_GAIN

from benchmark_fabric_leather import (Bank, ScratchModel, FEATURE_VERSION, lights40,
    update, prepare, unit_prediction, fixed_f0, ndrs_loss, render_batch, top_view_positions, write_json)
from train_fabric import cosine_lr, require_scheduler_ready
from live_training_plot import plateau


def atomic_checkpoint(path,payload):
    temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp)
    os.replace(temp,path)


def truncate_log_to_checkpoint(path,total_step):
    if not path.exists():return
    temp=path.with_name(path.name+'.tmp')
    with path.open('r',encoding='utf-8') as source,temp.open('w',encoding='utf-8') as target:
        for line in source:
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row['total_step']>total_step:break
            target.write(line)
    os.replace(temp,path)


def save_checkpoint(path,model,optimizer,rng,state,contract):
    atomic_checkpoint(path,{'format':'nfplight.fabric_leather.training.v2','net_g':model.net_g.state_dict(),
        'optimizer':optimizer.state_dict(),'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),
        'state':state,'contract':contract,'feature_version':FEATURE_VERSION,
        'normal_head':'xyz','normal_decoder':'unit normalization before loss/inference',
        'precision':'float32'})


@torch.no_grad()
def evaluate(model,bank,lights,material_limit=0):
    """Four fixed crops/material; material macro errors, pixel angular quantiles."""
    was_training=model.net_g.training
    model.net_g.eval()
    selected=list(range(len(bank.records)))
    if material_limit:
        selected=[]
        for category in ('Fabric','Leather'):
            selected.extend([i for i,r in enumerate(bank.records) if r['category']==category][:material_limit])
    rows=[]
    angles_by_category={'Fabric':[],'Leather':[]}
    start=time.perf_counter()
    try:
        for index in selected:
            crop_rows=[]
            angles=[]
            for crop in (0,4,8,12):
                target=bank.crop(index,crop)[None]
                safe,valid,features=prepare(model,target)
                pred=unit_prediction(model.net_g(features))
                maps,parts=ndrs_loss(pred,safe,valid)
                views=top_view_positions(40,device=model.device)
                a=render_batch(model,fixed_f0(pred),lights,views)
                b=render_batch(model,safe,lights,views)
                render=(a-b).abs().masked_select(valid[:,None].expand_as(a)).mean()
                dot=(pred[:,:3]*safe[:,:3]).sum(1,keepdim=True).clamp(-1,1)
                angle=torch.rad2deg(torch.acos(dot)).masked_select(valid).cpu().numpy()
                angles.append(angle)
                values=[float(x) for x in parts]+[float(render)]
                if not np.isfinite(values).all() or not np.isfinite(angle).all():
                    raise FloatingPointError('nonfinite validation values')
                crop_rows.append(values)
            angle=np.concatenate(angles)
            values=np.mean(crop_rows,axis=0)
            record=bank.records[index]
            row={'material_id':record['material_id'],'category':record['category'],'crops':4,
                **dict(zip(('map_normal','map_diffuse','map_roughness','map_specular','render_l1'),map(float,values))),
                'ndr_mean':float(np.mean(values[:3])),'ndrs_sum':float(np.sum(values[:4])),'normal_degrees':float(angle.mean()),
                'normal_degrees_median':float(np.median(angle)),'normal_degrees_p95':float(np.percentile(angle,95))}
            rows.append(row)
            angles_by_category[record['category']].append(angle)
    finally:
        model.net_g.train(was_training)
    keys=('map_normal','map_diffuse','map_roughness','map_specular','ndr_mean','ndrs_sum','render_l1','normal_degrees')
    macro={key:float(np.mean([r[key] for r in rows])) for key in keys}
    category={}
    all_angles=[]
    for name,arrays in angles_by_category.items():
        if not arrays:continue
        pixels=np.concatenate(arrays)
        all_angles.append(pixels)
        subset=[r for r in rows if r['category']==name]
        category[name]={'materials':len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in keys},
            'normal_degrees_pixel_median':float(np.median(pixels)),
            'normal_degrees_pixel_p95':float(np.percentile(pixels,95))}
    pixels=np.concatenate(all_angles)
    macro.update(normal_degrees_pixel_median=float(np.median(pixels)),normal_degrees_pixel_p95=float(np.percentile(pixels,95)))
    return {'mean':macro,'category':category,'materials':rows,'seconds':time.perf_counter()-start,
        'angular_quantiles':'all valid pixels in the fixed four crops; no material-median substitution',
        'crop_indices':[0,4,8,12],'official_test_payloads_read':0}


def contract_for(args,train,val):
    code_files=['train_fabric_leather.py','benchmark_fabric_leather.py','train_fabric.py',
                'data/fabric.py','model/nfplight_model.py','network/nfplight_net.py','utils/render_util.py']
    files=[Path(name).resolve() for name in code_files]
    files += [Path(args.manifest).resolve()]+[Path(r['shard_path']) for r in train.records+val.records]
    identities=[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files]
    return {'format':'nfplight.fabric_leather.contract.v2','map_steps':args.map_steps,'render_steps':args.render_steps,
        'batch':args.batch,'accumulate':args.accumulate,'val_every':args.val_every,'checkpoint_every':args.checkpoint_every,
        'seed':args.seed,'fp32':True,'amp':False,'tf32':False,'normal_loss':'unit XYZ L1',
        'map_reduction':'normal_xyz_component_mean + diffuse_component_mean + roughness_mean + specular_component_mean',
        'map_monitor':'ndrs_sum','diagnostic_ndr_mean':'reported only; not the optimized objective or best-map selection',
        'render_weight':1.,'f0_input_and_both_render_branches':.04,'specular_supervision':True,'specular_target':.04,
        'd4':'uniform rot90 plus hflip','renderer_grid':'R4 theta5,10,20,30,45 x phi0:45:315',
        'view_z':2.75,'gamma_diffuse':2.2,'roughness_floor':.05,'valid_mask':'GT normal z>0; opacity ignored',
        'input':{'near':INPUT_NEAR,'far':INPUT_FAR,'fixed_gain':INPUT_GAIN,'coordinates':'model units; surface [-1,1]',
                 'light_camera':'colocated','lamp_intensity':16,'encoding':'linear_rgb',
                 'quantize':False,'gamma':0,'denoiser':False},
        'feature_version':FEATURE_VERSION,'val_crops':[0,4,8,12],'validation_material_limit':args.validation_material_limit,
        'val_populations':{name:sum(r['category']==name for r in val.records) for name in ('Fabric','Leather')},
        'train_materials':len(train.records),'val_materials':len(val.records),'train_material_ids':[r['material_id'] for r in train.records],
        'manifest_text':Path(args.manifest).read_text(encoding='utf-8'),'file_identities':identities,
        'sha256_computed':False,'plateau':'per-stage 10 consecutive val intervals <.005 relative improvement; display only',
        'lr_map':[5e-4,1e-5],'lr_render':[1e-4,1e-5],'optimizer':'Adam betas(.9,.999), weight_decay0, clip_norm1',
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='C:/MatSynth_PRIMA_Fabric/manifests/fabric_leather_training_manifest.jsonl')
    parser.add_argument('--out',default='artifacts/fabric_leather_training_gain9_20261008')
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--accumulate',type=int,default=1)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--map-steps',type=int,default=400000)
    parser.add_argument('--render-steps',type=int,default=100000)
    parser.add_argument('--val-every',type=int,default=1000)
    parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--validation-material-limit',type=int,default=0,help='smoke only; per category')
    parser.add_argument('--stop-after',type=int,default=0,help='bounded integration check; does not change budget')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    require_scheduler_ready(historical_resume=args.resume, smoke=args.validation_material_limit > 0 and max(args.map_steps, args.render_steps) <= 10)
    if args.batch*args.accumulate!=4 or min(args.map_steps,args.render_steps,args.val_every,args.checkpoint_every)<=0:
        raise ValueError('positive schedules and effective batch4 required')
    if args.validation_material_limit and (args.map_steps>10 or args.render_steps>10):
        raise ValueError('partial validation allowed only for smoke schedules <=10+10')
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not args.resume and (out/'config.json').exists():raise FileExistsError('run exists; resume or use a new directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng=random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    train=Bank(args.manifest,'train');val=Bank(args.manifest,'val')
    assert len(train.records)==566 and len(val.records)==164
    assert not {f'{i}_fur_cow_long' for i in range(1652,1657)} & {r['material_id'] for r in train.records}
    contract=contract_for(args,train,val)
    model=ScratchModel(SimpleNamespace(device='cuda',precision='float32',feature_version=FEATURE_VERSION,capture_precision=True))
    lights=lights40(model.device)
    state={'stage':'map','step':0,'total_step':0,'history':[],'best_map':None,'best_render':None}
    optimizer=torch.optim.Adam(model.net_g.parameters(),lr=5e-4,betas=(.9,.999))
    if args.resume:
        checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['contract']!=contract:raise ValueError('resume contract/data/source metadata mismatch')
        model.net_g.load_state_dict(checkpoint['net_g'])
        state=checkpoint['state']
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.setstate(checkpoint['rng']);torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);np.random.set_state(checkpoint['numpy_rng'])
        truncate_log_to_checkpoint(out/'train.jsonl',state['total_step'])
        del checkpoint
    else:
        write_json(out/'config.json',contract)
    started=time.perf_counter()
    invocation_steps=0
    def status(kind,**extra):
        write_json(out/'status.json',{'status':kind,'pid':os.getpid(),'stage':state['stage'],'step':state['step'],
            'total_step':state['total_step'],'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'official_test_payloads_read':0,'sha256_computed':False,**extra})
    def validation():
        status('validating')
        print('validation start',state['stage'],state['step'],flush=True)
        latest=evaluate(model,val,lights,args.validation_material_limit)
        row={'stage':state['stage'],'step':state['step'],'total_step':state['total_step'],
             **latest['mean'],'category':latest['category']}
        state['history'].append(row)
        state['latest_validation']=latest
        summary={'history':state['history'],'latest':latest,
                 'plateau':{name:plateau(state['history'],name) for name in ('map','render')}}
        write_json(out/'validation.json',summary)
        metric=latest['mean']['ndrs_sum'] if state['stage']=='map' else latest['mean']['render_l1']
        key='best_map' if state['stage']=='map' else 'best_render'
        if state[key] is None or metric<state[key]:
            state[key]=metric
            save_checkpoint(out/(key+'.pt'),model,optimizer,rng,state,contract)
        print('validation done',state['total_step'],latest['mean'],flush=True)
    try:
        status('initializing')
        print('train566 val164, FP32, batch',args.batch,'accumulate',args.accumulate,'pid',os.getpid(),flush=True)
        # A step-zero full validation makes the live view interpretable before the first 1000-step evaluation.
        if not state['history']:validation()
        write_json(out/'validation.json',{'history':state['history'],'latest':state.get('latest_validation'),'plateau':{name:plateau(state['history'],name) for name in ('map','render')}})
        save_checkpoint(out/'last.pt',model,optimizer,rng,state,contract)
        status('training')
        with (out/'train.jsonl').open('a',encoding='utf-8') as stream:
            while state['stage']!='complete':
                limit=args.map_steps if state['stage']=='map' else args.render_steps
                if state['step']>=limit:
                    if state['stage']=='map':
                        save_checkpoint(out/'stage_a.pt',model,optimizer,rng,state,contract)
                        state.update(stage='render',step=0)
                        optimizer=torch.optim.Adam(model.net_g.parameters(),lr=1e-4,betas=(.9,.999))
                        validation()
                        save_checkpoint(out/'last.pt',model,optimizer,rng,state,contract)
                        status('training')
                        continue
                    state['stage']='complete'
                    break
                lr=cosine_lr(state['step'],limit,5e-4 if state['stage']=='map' else 1e-4)
                for group in optimizer.param_groups:group['lr']=lr
                tick=time.perf_counter()
                values=update(model,train,optimizer,rng,None if state['stage']=='map' else lights,
                    batch=args.batch,accumulate=args.accumulate)
                state['step']+=1;state['total_step']+=1;invocation_steps+=1
                row={'stage':state['stage'],'step':state['step'],'total_step':state['total_step'],'lr':lr,
                     'seconds':time.perf_counter()-tick,'render_weight':0 if state['stage']=='map' else 1,
                     'loss':values['map_loss']+values['render_loss'],**values}
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if state['total_step']==1 or state['total_step']%100==0:
                    stream.flush();status('training',latest_loss=row['loss'],latest_seconds=row['seconds'])
                    print('update',state['total_step'],state['stage'],round(row['loss'],6),'seconds',round(row['seconds'],3),flush=True)
                val_due=state['step']%args.val_every==0 or state['step']==limit
                checkpoint_due=state['step']%args.checkpoint_every==0
                if val_due:
                    stream.flush();validation();status('training')
                if val_due or checkpoint_due:
                    stream.flush();save_checkpoint(out/'last.pt',model,optimizer,rng,state,contract)
                if args.stop_after and invocation_steps>=args.stop_after:
                    stream.flush();save_checkpoint(out/'last.pt',model,optimizer,rng,state,contract)
                    status('bounded_stop');return
            stream.flush()
        save_checkpoint(out/'last.pt',model,optimizer,rng,state,contract)
        status('complete')
    except (Exception,KeyboardInterrupt) as exc:
        status('failed',error=repr(exc))
        write_json(out/'failure.json',{'error':repr(exc),'state':{k:v for k,v in state.items() if k!='history'}})
        # Keep the last fully committed checkpoint, never replace it with a partial update.
        raise


if __name__=='__main__':main()
