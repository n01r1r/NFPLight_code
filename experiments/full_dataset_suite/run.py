"""Streaming full-bank FP32 runner for automatically authorized arms1-3."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from model.nfplight_model import NFPLightModel
from benchmark_fabric_leather import Bank, prepare as prepare_features, unit_prediction, fixed_f0, ndrs_loss
from train_fabric import render_batch, top_view_positions
from experiments.bottleneck_fc.pipeline import make_lights
from experiments.regularization.run import _seed
from experiments.regularization.config import atomic_write_json
from experiments.robustness_suite.run import create_network, process_lock, memory_observation, memory_is_low, runtime_environment
from experiments.full_dataset_suite.config import DEFAULT_OUT, ARMS, prepare
from experiments.full_dataset_suite.engine import Engine, neutral_evaluation, load_checkpoint
from experiments.full_dataset_suite.storage import require_disk_reserve

class Model(NFPLightModel):
    def __init__(self,policy,backbone):
        self.policy,self.backbone=policy,backbone
        super().__init__(SimpleNamespace(device="cuda",precision="float32",feature_version="raw_calibrated_gain9_v3",capture_precision=True))
    def init_network(self):
        self.net_g=self.model_to_device(create_network(self.policy,self.backbone))

def render_loss(model,pred,target,valid,lights):
    views=top_view_positions(len(lights),device=model.device)
    with torch.no_grad(): reference=render_batch(model,target,lights,views)
    prediction=render_batch(model,fixed_f0(pred),lights,views)
    return (prediction-reference).abs().masked_select(valid[:,None].expand_as(prediction)).mean()

class Pipeline:
    def __init__(self,execution,*,smoke=False):
        self.execution=execution;self.smoke=smoke
        torch.set_num_threads(4);_seed(execution["policy"]["seed"])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        self.model=Model(execution["policy"],execution["scientific_settings"]["backbone"])
        self.train=Bank(execution["manifest"],"train");self.val=Bank(execution["manifest"],"val")
        if self.train.records!=execution["train"] or self.val.records!=execution["validation"]:
            raise ValueError("bank sample folders differ from immutable contract")
        r=execution["scientific_settings"]["rendering"]
        self.training_lights=make_lights(r["training_lights"],self.model.device)
        self.novel_lights=make_lights(r["novel_lights"],self.model.device)
        if len(self.training_lights)!=40 or len(self.novel_lights)!=12:raise ValueError("light populations differ")
    def objective(self,network,indices,transforms,stage):
        target=torch.stack([self.train.crop(index//16,index%16,transform%4,transform>=4)
                            for index,transform in zip(indices,transforms)])
        safe,valid,features=prepare_features(self.model,target)
        pred=unit_prediction(network(features))
        maps,parts=ndrs_loss(pred,safe,valid)
        rendered=render_loss(self.model,pred,safe,valid,self.training_lights) if stage=="render" else maps.new_zeros(())
        return maps+rendered,{"ndrs_sum":float(maps.detach()),"ndr_mean":float(sum(parts[:3]).detach()/3),"map_loss":float(maps.detach()),"render_loss":float(rendered.detach()),
            **dict(zip(("normal_loss","diffuse_loss","roughness_loss","specular_loss"),[float(x.detach()) for x in parts])),
            "map_aggregation":"ndrs_sum"}
    def evaluate(self,network,stage,step):
        started=time.perf_counter();rows=[]
        selected=list(range(len(self.val.records)))
        if self.smoke:
            selected=[next(i for i,r in enumerate(self.val.records) if r["category"]==category) for category in ("Fabric","Leather")]
        for index in selected:
            crops=[]
            for crop in (0,4,8,12):
                safe,valid,features=prepare_features(self.model,self.val.crop(index,crop)[None])
                pred=unit_prediction(network(features))
                _,parts=ndrs_loss(pred,safe,valid)
                rendered=render_loss(self.model,pred,safe,valid,self.novel_lights)
                angle=torch.rad2deg(torch.acos((pred[:,:3]*safe[:,:3]).sum(1,keepdim=True).clamp(-1,1))).masked_select(valid).mean()
                values=[float(x) for x in parts]+[float(rendered),float(angle)]
                if not np.isfinite(values).all():raise FloatingPointError("nonfinite validation")
                crops.append(values)
            values=np.mean(crops,axis=0)
            record=self.val.records[index]
            rows.append({"material_id":record["material_id"],"category":record["category"],"group_id":record["group_id"],
                **dict(zip(("map_normal","map_diffuse","map_roughness","map_specular","render_l1","normal_degrees"),map(float,values))),
                "ndr_mean":float(np.mean(values[:3])),"ndrs_sum":float(sum(values[:4]))})
        keys=("map_normal","map_diffuse","map_roughness","map_specular","render_l1","normal_degrees","ndr_mean","ndrs_sum")
        def mean(subset):return {k:float(np.mean([row[k] for row in subset])) for k in keys}
        category={name:{"materials":sum(r["category"]==name for r in rows),**mean([r for r in rows if r["category"]==name])} for name in ("Fabric","Leather")}
        groups={g:mean([r for r in rows if r["group_id"]==g]) for g in sorted({r["group_id"] for r in rows})}
        heldout={"mean":mean(rows),"category":category,"materials":rows,
                 "source_group_macro":{k:float(np.mean([row[k] for row in groups.values()])) for k in keys},
                 "sample_folders":len(rows),"source_groups":len(groups),"seconds":time.perf_counter()-started,
                 "crop_indices":[0,4,8,12],"novel_lights":12,"official_test_payloads_read":0}
        print("validation",stage,step,heldout["sample_folders"],heldout["mean"],"seconds",heldout["seconds"],flush=True)
        return {"heldout":heldout}

def make_engine(output,execution,*,smoke=False):
    pipeline=Pipeline(execution,smoke=smoke)
    engine=Engine(output,pipeline.model.net_g,execution["policy"],execution,pipeline.objective,pipeline.evaluate,sample_count=9056)
    engine.memory_guard=lambda:memory_is_low(memory_observation())
    return pipeline,engine

def run(output,arm,*,resume=False):
    execution=prepare(output,arm);require_disk_reserve(output)
    with process_lock(output,resume=resume) as process:
        environment=runtime_environment()
        runtime_path=Path(output)/"runtime.json"
        if runtime_path.exists() and json.loads(runtime_path.read_text())["environment"]!=environment:
            raise ValueError("runtime environment differs")
        atomic_write_json(runtime_path,{"environment":environment,"process":process})
        pipeline,engine=make_engine(output,execution)
        result=engine.run({k:execution["policy"][k] for k in ("map_cap","render_cap")},resume=resume)
        if result["status"]=="complete":
            selected=torch.load(Path(output)/"selected.pt",map_location="cpu",weights_only=True)
            require_disk_reserve(output)
            from experiments.full_dataset_suite.engine import atomic_save
            atomic_save(Path(output)/"net_g_selected_fp32.pth",{"net_g":selected["net_g"],"feature_version":"raw_calibrated_gain9_v3","precision":"float32","normal_decoder":"unit XYZ"})
        return result

def smoke(output,arm):
    execution=prepare(output,arm)
    pipeline,engine=make_engine(output,execution,smoke=True)
    caps={"map_cap":400000,"render_cap":100000}
    start=time.perf_counter()
    engine.run(caps,interrupt_after=2)
    engine.transition()
    engine.run(caps,resume=True,interrupt_after=4)
    before=engine.snapshot()
    engine.restore(load_checkpoint(Path(output)/"last.pt"))
    if engine.state["total_step"]!=before["state"]["total_step"] or engine.scheduler.state_dict()!=before["scheduler"]:
        raise AssertionError("smoke restore state differs")
    for key,value in engine.net.state_dict().items():
        if not torch.equal(value.cpu(),before["net_g"][key]):raise AssertionError("smoke restored weights differ")
    torch.cuda.synchronize()
    atomic_write_json(Path(output)/"smoke_verification.json",{"passed":True,"arm":arm,"map_updates":2,"render_updates":2,
        "train_sample_folders":566,"train_slots":9056,"validation_sample_folders_smoke_only":2,
        "novel_lights":12,"training_lights":40,"elapsed_seconds":time.perf_counter()-start,
        "source_input_identity":execution["file_identities"],"policy":execution["policy"],
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"precision":"float32","amp":False,"tf32":False,"sha256_computed":False})

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("prepare","run","smoke"));p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--resume",action="store_true");a=p.parse_args()
    if a.mode=="prepare":prepare(a.out,a.arm)
    elif a.mode=="smoke":smoke(a.out,a.arm)
    else:run(a.out,a.arm,resume=a.resume)

if __name__=="__main__":main()
