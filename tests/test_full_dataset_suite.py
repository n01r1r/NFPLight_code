import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.full_dataset_suite import config
from experiments.full_dataset_suite.engine import Engine,metric_values,load_checkpoint
from experiments.full_dataset_suite.controller import authorized_jobs
from experiments.full_dataset_suite import storage
from experiments.robustness_suite.run import create_network


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def build(out):
    torch.manual_seed(19);np.random.seed(19);random.seed(19)
    policy=config.policy_for_arm(config.ARMS[0])
    policy.update(map_lr=.01,map_min_lr=.0025,render_lr=.01,render_min_lr=.0025,
                  patience=0,cooldown=0,floor_observations=2,evaluation_interval=1,checkpoint_interval=1)
    net=nn.Sequential(nn.Linear(2,1),nn.Dropout(.1))
    def objective(net,indices,transforms,stage):
        features=torch.tensor([[sum(indices)/20,sum(transforms)/32]])
        return (net(features)-.8).square().mean(),{}
    def evaluate(net,stage,step):
        assert not net.training and not torch.is_grad_enabled()
        torch.rand(3);random.random();np.random.rand()
        return {"heldout":{"mean":{"ndrs_sum":1 if step==0 else .9 if step==1 else .95,
            "ndr_mean":.01/(step+1),"render_l1":1 if stage=="map" or step==0 else .5 if step==1 else .51}}}
    return Engine(out,net,policy,{"fixture":True},objective,evaluate,sample_count=8)

@pytest.mark.parametrize("cut",[3,5,8])
def test_exact_full_engine_resume_and_ndrs_selection(tmp_path,cut):
    caps={"map_cap":30,"render_cap":30}
    a=build(tmp_path/"a");a.run(caps)
    b=build(tmp_path/"b");b.run(caps,interrupt_after=cut)
    c=build(tmp_path/"b");c.run(caps,resume=True)
    equal(a.snapshot(),c.snapshot())
    assert c.state["best_map"]["step"]==1
    assert c.state["best_map"]["ndrs_sum"]==.9
    assert load_checkpoint(tmp_path/"b/last.pt")["package"]=="full_dataset_suite"
    assert len(list((tmp_path/"b").glob("*.pt")))==7
    assert c.seen_slots


def test_caps_pause_without_forced_render_and_selection(tmp_path):
    e=build(tmp_path/"run");e.run({"map_cap":2,"render_cap":2})
    assert e.state["status"]=="budget_paused"
    assert e.state["stage"]=="map" and e.state["render_updates"]==0
    assert not (tmp_path/"run/selected.pt").exists()


def test_ndrs_sum_monitor_and_invalid_results():
    assert metric_values({"heldout":{"mean":{"ndrs_sum":2.,"ndr_mean":.001,"render_l1":.1}}})==(2.,.1)
    with pytest.raises(FloatingPointError):metric_values({"heldout":{"mean":{"ndrs_sum":float("nan"),"render_l1":.1}}})

@pytest.mark.parametrize("arm,params,p",[(config.ARMS[0],7609834,.1),(config.ARMS[1],15085034,.1),(config.ARMS[2],7609834,.5)])
def test_authorized_depth_dropout_and_adequate_caps(arm,params,p):
    policy=config.policy_for_arm(arm)
    net=create_network(policy,{"width":32,"encoder_blocks":[2,2,4],"decoder_blocks":[2,2,2],"middle_blocks":policy["middle_blocks"]})
    assert sum(x.numel() for x in net.parameters())==params
    assert all(m.p==p for m in net.modules() if isinstance(m,nn.Dropout))
    assert policy["map_cap"]==400000 and policy["render_cap"]==100000
    assert policy["evaluation_interval"]==1000 and policy["ema_decay"] is None


def test_full_sample_folder_scope_and_queue_restriction():
    train,val=config.dataset_rows()
    assert len(train)*16==9056 and len(val)==164
    assert len([r for r in train if "fabric_suit_vintage" in r["material_id"]])==21
    assert authorized_jobs({"arms":list(config.ARMS)})==list(config.ARMS)
    with pytest.raises(ValueError):authorized_jobs({"arms":["ema7m_p01"]})
    with pytest.raises(ValueError):config.policy_for_arm("extra_crop7m_p01")


def test_disk_reserve_prevents_write(monkeypatch,tmp_path):
    monkeypatch.setattr(storage.shutil,"disk_usage",lambda _:SimpleNamespace(free=1024))
    with pytest.raises(storage.DiskReserveError):storage.require_disk_reserve(tmp_path)
    assert not list(tmp_path.iterdir())


def test_streaming_all_folder_crop_indices_and_d4(monkeypatch):
    from experiments.full_dataset_suite import run
    pipeline=run.Pipeline.__new__(run.Pipeline)
    calls=[]
    def crop(i,c,r,f):
        calls.append((i,c,r,f))
        t=torch.zeros(10,4,4);t[2]=1;t[7:]=-.92
        return t
    pipeline.train=SimpleNamespace(crop=crop);pipeline.model=SimpleNamespace(device="cpu")
    monkeypatch.setattr(run,"prepare_features",lambda model,t:(t,torch.ones(t.shape[0],1,4,4,dtype=torch.bool),t))
    loss,metrics=pipeline.objective(nn.Identity(),[0,9055],[0,7],"map")
    assert calls==[(0,0,0,False),(565,15,3,True)]
    assert metrics["map_aggregation"]=="ndrs_sum"
    assert metrics["specular_loss"]==0 and float(loss)==0


def test_validation_novel_lights_and_folder_group_macros(monkeypatch):
    from experiments.full_dataset_suite import run
    pipeline=run.Pipeline.__new__(run.Pipeline);pipeline.smoke=False
    records=[{"material_id":"a","category":"Fabric","group_id":"g"},
             {"material_id":"b","category":"Fabric","group_id":"g"},
             {"material_id":"c","category":"Leather","group_id":"h"}]
    def crop(i,c):
        t=torch.zeros(10,4,4);t[2]=1;t[3:6]=[0.,.2,.8][i];t[7:]=-.92
        return t
    pipeline.val=SimpleNamespace(records=records,crop=crop)
    pipeline.model=SimpleNamespace(device="cpu");pipeline.novel_lights=torch.zeros(12,3)
    monkeypatch.setattr(run,"prepare_features",lambda model,t:(t,torch.ones(t.shape[0],1,4,4,dtype=torch.bool),t))
    def fake_render(model,pred,target,valid,lights):
        assert len(lights)==12
        return pred.new_zeros(())
    monkeypatch.setattr(run,"render_loss",fake_render)
    class Network(nn.Module):
        def forward(self,t):
            out=t.clone();out[:,3:6]=0
            return out
    with torch.no_grad():r=pipeline.evaluate(Network(),"map",0)["heldout"]
    assert r["sample_folders"]==3 and r["source_groups"]==2
    assert r["mean"]["ndrs_sum"]==pytest.approx(1/3)
    assert r["source_group_macro"]["ndrs_sum"]==pytest.approx(.45)


def test_low_disk_rolls_back_to_committed_update(monkeypatch,tmp_path):
    e=build(tmp_path/"run")
    real=storage.shutil.disk_usage
    monkeypatch.setattr(storage.shutil,"disk_usage",lambda path:SimpleNamespace(free=1024) if e.state["total_step"]>=2 else real(path))
    e.run({"map_cap":30,"render_cap":30})
    assert e.state["status"]=="disk_paused" and e.state["total_step"]==1
    assert e.committed_step==1
    assert len((tmp_path/"run/train.jsonl").read_text().splitlines())==1
