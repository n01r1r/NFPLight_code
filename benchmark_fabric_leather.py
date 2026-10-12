"""Bounded FP32 speed/sanity probe for the agreed Fabric+Leather contract.
No production checkpoints, official-test evaluation, or hash verification.
"""
import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from model.nfplight_model import INPUT_NEAR, INPUT_FAR, INPUT_GAIN
from torch.nn import functional as F
from data.fabric import pack_svbrdf, canonical_target, write_json
from train_fabric import ScratchModel, FEATURE_VERSION, render_input_images_batched, render_batch, masked_l1, top_view_positions

MAPS = ('normal','diffuse','roughness','specular')


def fixed_f0(maps):
    return torch.cat((maps[:,:7], torch.full_like(maps[:,7:10], 2*.04-1)),1)


def unit_prediction(raw):
    if not torch.isfinite(raw).all() or (raw[:,:3].norm(dim=1)<1e-8).any():
        raise FloatingPointError('invalid predicted normal')
    return torch.cat((F.normalize(raw[:,:3],dim=1),raw[:,3:]),1)


def lights40(device):
    return torch.tensor([[4*math.sin(math.radians(t))*math.cos(math.radians(p)),
                          4*math.sin(math.radians(t))*math.sin(math.radians(p)),
                          4*math.cos(math.radians(t))] for t in (5,10,20,30,45) for p in range(0,360,45)],dtype=torch.float32,device=device)


class Bank:
    def __init__(self,manifest,split):
        self.records=[json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        self.records=[row for row in self.records if row['split']==split]
        self.cache=OrderedDict()
        self.loads=0

    def crop(self,index,crop,rotation=0,flip=False):
        row=self.records[index]
        path=row['shard_path']
        if path not in self.cache:
            with np.load(path,allow_pickle=False) as archive:
                maps={key:archive[key] for key in MAPS}
            if any(value.dtype!=np.uint8 for value in maps.values()):
                raise ValueError('bank must use uint8 maps')
            self.cache[path]=maps
            self.loads+=1
            if len(self.cache)>4:
                self.cache.popitem(last=False)
        maps=self.cache[path]
        self.cache.move_to_end(path)
        normal=maps['normal'][crop].astype(np.float32)/127.5-1
        length=np.linalg.norm(normal,axis=-1,keepdims=True)
        if (length<.01).any():
            raise ValueError('invalid source normal')
        diffuse=(maps['diffuse'][crop].astype(np.float32)/255)**2.2
        rough=maps['roughness'][crop].astype(np.float32)/255
        if rough.ndim==2:
            rough=rough[...,None]
        # Preserved specular is loaded as required bank field, but not decoded/used.
        spec=np.full_like(diffuse,.04)
        return pack_svbrdf(normal/length,diffuse,rough,spec,rot=rotation,hflip=flip)

    def sample(self,rng,count,fixed=None):
        slots=fixed or [(rng.randrange(len(self.records)),rng.randrange(16),rng.randrange(4),bool(rng.randrange(2))) for _ in range(count)]
        return torch.stack([self.crop(*slot) for slot in slots]),slots


def prepare(model,target):
    safe,valid=canonical_target(target.to(model.device))
    safe=fixed_f0(safe)
    with torch.no_grad():
        observations=render_input_images_batched(model,safe)*valid
        features=model.build_features(observations,valid_mask=valid)
    return safe,valid,features[0] if isinstance(features,tuple) else features


def ndr_loss(pred,target,mask):
    parts=[masked_l1(pred[:,sl],target[:,sl],mask,validate=False) for sl in (slice(0,3),slice(3,6),slice(6,7))]
    return sum(parts)/3,parts


def ndrs_loss(pred, target, mask):
    """Sum four per-map L1 terms; normal is unit-normalized by the caller.

    Specular supervision targets the existing fixed F0=.04 bank value.
    The historical ndr_loss above remains an evaluation diagnostic only.
    """
    parts = [masked_l1(pred[:, section], target[:, section], mask)
             for section in (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10))]
    return sum(parts), parts


def sync():
    torch.cuda.synchronize()


def update(model,bank,optimizer,rng,lights,*,batch=2,accumulate=2,fixed=None,profile=False):
    optimizer.zero_grad(set_to_none=True)
    times={}
    stats=[]
    slots_all=[]
    def measured(name,fn):
        if profile:sync()
        start=time.perf_counter()
        result=fn()
        if profile:sync()
        times[name]=times.get(name,0)+time.perf_counter()-start
        return result
    for _ in range(accumulate):
        target,slots=measured('load_decode_d4',lambda:bank.sample(rng,batch,fixed))
        slots_all.extend(slots)
        safe,valid,features=measured('transfer_input_features',lambda:prepare(model,target))
        raw=measured('network_forward',lambda:model.net_g(features))
        raw.retain_grad()
        pred=unit_prediction(raw)
        maps,parts=measured('map_loss',lambda:ndrs_loss(pred,safe,valid))
        rendered=maps.new_zeros(())
        if lights is not None:
            views=top_view_positions(len(lights),device=model.device)
            def rendering():
                with torch.no_grad():
                    gt=render_batch(model,safe,lights,views)
                predicted=render_batch(model,fixed_f0(pred),lights,views)
                mask=valid[:,None].expand(-1,len(lights),3,-1,-1)
                return (predicted-gt).abs().masked_select(mask).mean()
            rendered=measured('render_forward',rendering)
        loss=(maps+rendered)/accumulate
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        measured('backward',loss.backward)
        stats.append([*[float(x.detach()) for x in parts],float(rendered.detach())])
    norm=measured('clip_and_optimizer',lambda:step_optimizer(model,optimizer))
    values=np.mean(np.asarray(stats),axis=0)
    return {'map_loss':float(values[:4].sum()), 'normal_loss':float(values[0]),
            'diffuse_loss':float(values[1]), 'roughness_loss':float(values[2]),
            'specular_loss':float(values[3]), 'render_loss':float(values[4]),
            'map_aggregation':'ndrs_sum', 'preclip_grad_norm':norm,
            'slots':slots_all,'phases_seconds':times if profile else {}}


def step_optimizer(model,optimizer):
    norm=torch.nn.utils.clip_grad_norm_(model.net_g.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm<=0:
        raise FloatingPointError('zero or nonfinite parameter gradient')
    optimizer.step()
    return float(norm)


def measure_case(model,bank,rng,out,label,lights,steps,batch,accumulate):
    optimizer=torch.optim.Adam(model.net_g.parameters(),lr=5e-4 if lights is None else 1e-4,betas=(.9,.999))
    for _ in range(2):
        update(model,bank,optimizer,rng,lights,batch=batch,accumulate=accumulate)
    sync()
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    start=time.perf_counter()
    for i in range(steps):
        sync()
        tick=time.perf_counter()
        values=update(model,bank,optimizer,rng,lights,batch=batch,accumulate=accumulate)
        sync()
        values.update(step=i+1,stage=label,total_step=i+1,seconds=time.perf_counter()-tick)
        rows.append(values)
        print(label,i+1,round(values['seconds'],3),round(values['map_loss'],5),flush=True)
    seconds=time.perf_counter()-start
    memory={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    profile=update(model,bank,optimizer,rng,lights,batch=batch,accumulate=accumulate,profile=True)
    result={'steps':steps,'seconds':seconds,'mean_seconds_per_update':statistics.mean(row['seconds'] for row in rows),
            'median_seconds_per_update':statistics.median(row['seconds'] for row in rows),
            'p90_seconds_per_update':float(np.percentile([row['seconds'] for row in rows],90)),
            'memory_bytes':memory,'profile':profile,'rows':rows}
    write_json(out/(label+'.json'),result)
    return result


@torch.no_grad()
def evaluate_probe(model,bank,lights,out):
    selected=[]
    for category in ('Fabric','Leather'):
        selected.extend([i for i,row in enumerate(bank.records) if row['category']==category][:4])
    model.net_g.eval()
    rows=[]
    sync()
    start=time.perf_counter()
    for index in selected:
        for crop in (0,4,8,12):
            target=bank.crop(index,crop)[None]
            safe,valid,features=prepare(model,target)
            pred=unit_prediction(model.net_g(features))
            maps,parts=ndr_loss(pred,safe,valid)
            views=top_view_positions(len(lights),device=model.device)
            a=render_batch(model,fixed_f0(pred),lights,views)
            b=render_batch(model,safe,lights,views)
            rows.append({'material_id':bank.records[index]['material_id'],'category':bank.records[index]['category'],'crop':crop,
                         'ndr_mean':float(maps),'render_l1':float((a-b).abs().masked_select(valid[:,None].expand_as(a)).mean())})
    sync()
    elapsed=time.perf_counter()-start
    model.net_g.train()
    result={'materials':len(selected),'crops_per_material':4,'seconds':elapsed,'rows':rows,
            'full_val_seconds_extrapolated':elapsed*len(bank.records)/len(selected),
            'performance_probe_only':True,'not_checkpoint_selection':True}
    write_json(out/'validation_probe.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='C:/MatSynth_PRIMA_Fabric/manifests/fabric_leather_training_manifest.jsonl')
    parser.add_argument('--out',default='artifacts/fabric_leather_speed_20261007')
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--sanity-steps',type=int,default=24)
    parser.add_argument('--support-audit',type=Path,help='Explicitly restrict this SPEED probe to nonzero-support materials; never edits manifest')
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--accumulate',type=int,default=2)
    args=parser.parse_args()
    if args.batch*args.accumulate!=4:
        raise ValueError('agreed effective batch is four')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('preserve existing results; choose new output')
    torch.manual_seed(7)
    np.random.seed(7)
    rng=random.Random(7)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)
    train=Bank(args.manifest,'train')
    val=Bank(args.manifest,'val')
    exclusions=[]
    if args.support_audit:
        audit=json.loads(args.support_audit.read_text(encoding='utf-8'))
        exclusions=audit['invalid_materials']
        bad={row['material_id'] for row in exclusions}
        train.records=[row for row in train.records if row['material_id'] not in bad]
        val.records=[row for row in val.records if row['material_id'] not in bad]
        write_json(out/'speed_probe_exclusions.json',{'excluded':exclusions,'benchmark_only':True,'manifest_unchanged':True})
    model=ScratchModel(SimpleNamespace(device='cuda',precision='float32',feature_version=FEATURE_VERSION,capture_precision=True))
    lights=lights40(model.device)
    write_json(out/'config.json',{'seed':7,'dtype':'float32','tf32':False,'amp':False,'feature_version':FEATURE_VERSION,'input_near':INPUT_NEAR,'input_far':INPUT_FAR,'fixed_gain':INPUT_GAIN,
                                'manifest':args.manifest,'train_materials':len(train.records),'val_materials':len(val.records),
                                'normal':'unit XYZ L1','map_reduction':'sum of normal/diffuse/roughness/specular terms','specular_supervision':True,'specular_target':.04,
                                'f0_input_and_both_loss':.04,'render_weight':1.,'lights40':lights.cpu().tolist(),
                                'batch':args.batch,'accumulate':args.accumulate,'d4':'uniform','cache':'LRU 4 uint8 materials',
                                'roughness_floor':.05,'mask':'positive-z GT normals; opacity ignored','gamma':'diffuse gamma2.2 inherited',
                                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                                'sha256_computed':False,'production_training_started':False,'speed_probe_excluded_materials':[row['material_id'] for row in exclusions],
                                'full_dataset_training_gate_passed':not exclusions})
    results={}
    try:
        for label,probes in [('map',None),('render10',lights[::4]),('render20',lights[::2]),('render40',lights)]:
            results[label]=measure_case(model,train,rng,out,label,probes,args.steps,args.batch,args.accumulate)
        print('validation speed probe',flush=True)
        validation=evaluate_probe(model,val,lights,out)
        # Training implementation check: fresh random initialization, repeated one Fabric+one Leather batch.
        del model
        torch.cuda.empty_cache()
        torch.manual_seed(7)
        model=ScratchModel(SimpleNamespace(device='cuda',precision='float32',feature_version=FEATURE_VERSION,capture_precision=True))
        fixed=[(next(i for i,r in enumerate(train.records) if r['category']==category),0,0,False) for category in ('Fabric','Leather')]
        if args.batch!=2:
            fixed=[fixed[i%2] for i in range(args.batch)]
        opt=torch.optim.Adam(model.net_g.parameters(),lr=5e-4)
        sanity=[]
        for step in range(args.sanity_steps):
            row=update(model,train,opt,rng,None,batch=args.batch,accumulate=args.accumulate,fixed=fixed)
            row.update(total_step=step+1,step=step+1,stage='map',loss=row['map_loss'],lr=5e-4)
            sanity.append(row)
            with (out/'train.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            print('sanity',step+1,round(row['map_loss'],5),flush=True)
        first=statistics.mean(row['map_loss'] for row in sanity[:3])
        last=statistics.mean(row['map_loss'] for row in sanity[-3:])
        write_json(out/'sanity.json',{'first3_mean':first,'last3_mean':last,'relative_decrease':(first-last)/first,
                                   'loss_decreased':last<first,'rows':sanity,'is_overfit_check_not_generalization':True})
        map_sec=results['map']['mean_seconds_per_update']
        render_sec=results['render40']['mean_seconds_per_update']
        train_seconds=400000*map_sec+100000*render_sec
        validation_seconds=500*validation['full_val_seconds_extrapolated']
        report={'status':'complete_speed_probe','full_dataset_training_gate_passed':not exclusions,'excluded_materials':[row['material_id'] for row in exclusions],'cases':{key:{k:v for k,v in value.items() if k!='rows'} for key,value in results.items()},
                'validation':{key:value for key,value in validation.items() if key!='rows'},
                'sanity':{'first3_mean':first,'last3_mean':last,'loss_decreased':last<first},
                'eta':{'map_hours':400000*map_sec/3600,'render_hours':100000*render_sec/3600,
                       'train_hours':train_seconds/3600,'validation_hours_extrapolated':validation_seconds/3600,
                       'total_hours_excluding_checkpoint_and_startup':(train_seconds+validation_seconds)/3600,
                       'range_heuristic_20percent':[.8*(train_seconds+validation_seconds)/3600,1.2*(train_seconds+validation_seconds)/3600],
                       'not_statistical_confidence_interval':True},'sha256_computed':False,'production_training_started':False}
        write_json(out/'summary.json',report)
        print(json.dumps(report['eta']),flush=True)
    except Exception as exc:
        write_json(out/'failure.json',{'type':type(exc).__name__,'error':str(exc),'completed_cases':list(results)})
        raise


if __name__=='__main__':
    main()
