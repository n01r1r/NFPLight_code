import copy,random
import pytest,torch
from experiments.render_factor_followup import RenderEngine,identity,PARENT,read
from experiments.robustness_suite.engine import capture_rng

def fixture(tmp_path):
 policy=copy.deepcopy(read(PARENT/"execution_config.json")["policy"]);policy.update(factor=.8,evaluation_interval=1,checkpoint_interval=1)
 net=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(.1))
 def objective(net,indices,transforms,stage):
  assert stage=="render"
  return net(torch.ones(4,1)).square().mean(),{}
 def evaluate(net,stage,step):return {"heldout":{"mean":{"ndr_mean":1.,"render_l1":1.}}}
 eng=RenderEngine(tmp_path,net,policy,{"test":True},objective,evaluate,sample_count=24)
 parent=eng.snapshot();parent["state"]["map_updates"]=7100;parent["state"]["total_step"]=7100
 parent["state"]["best_map"]={"net_g":copy.deepcopy(net.state_dict()),"ndr_mean":1.,"render_l1":1.,"stage":"map","step":500,"total_step":500,"metric":1.}
 return eng,parent

def test_identity_only_factor_scientific_change(tmp_path,monkeypatch):
 import experiments.render_factor_followup as followup
 import json
 parent=tmp_path / "parent";parent.mkdir()
 old={"policy":{"arm":"capacity15m_p01","factor":.5,"render_cap":15000,"render_lr":1e-4},"scientific_settings":{"feature_version":"raw_calibrated_gain9_v3"},"file_identities":[]}
 (parent / "execution_config.json").write_text(json.dumps(old),encoding="utf8")
 (parent / "status.json").write_text('{"status":"complete"}',encoding="utf8")
 for name in ("best_map.pt","map_stage_end.pt"):(parent / name).write_bytes(b'fixture identity only; not loaded')
 monkeypatch.setattr(followup,"PARENT",parent)
 ex=identity()
 assert {k for k in ex["policy"] if ex["policy"][k]!=old["policy"][k]}=={"arm","factor","render_cap"}
 assert ex["scientific_settings"]==old["scientific_settings"]
 assert ex["policy"]["render_lr"]==1e-4

def test_canonical_initial_and_resume(tmp_path):
 eng,parent=fixture(tmp_path);eng.initialize_render(parent)
 assert eng.state["stage"]=="render" and eng.state["stage_step"]==0
 assert not eng.optimizer.state and eng.samples.getstate()==parent["sample_rng"]
 assert eng.scheduler.factor==.8 and eng.optimizer.param_groups[0]["lr"]==1e-4
 eng.evaluate();eng.committed_step=0;eng.commit();saved=eng.snapshot()
 eng.run({"map_cap":50000,"render_cap":2},resume=True)
 assert eng.state["status"]=="budget_paused"
 expected=copy.deepcopy(eng.net.state_dict());expected_rows=(tmp_path/"train.jsonl").read_text()
 eng.restore(saved);eng.committed_step=7100;eng.commit()
 eng.run({"map_cap":50000,"render_cap":2},resume=True)
 assert all(torch.equal(v,eng.net.state_dict()[k]) for k,v in expected.items())
 import json
 rows=[json.loads(x) for x in (tmp_path/"train.jsonl").read_text().splitlines()]
 original=[json.loads(x) for x in expected_rows.splitlines()]
 assert [(x['indices'],x['d4_transforms'],x['loss']) for x in rows]==[(x['indices'],x['d4_transforms'],x['loss']) for x in original]

def test_factor08_plateau_schedule_and_guard(tmp_path):
 eng,parent=fixture(tmp_path);eng.initialize_render(parent)
 for _ in range(7):eng.evaluate()
 assert eng.optimizer.param_groups[0]["lr"]==pytest.approx(8e-5)
 assert len(eng.state["history"])==7
 with pytest.raises(RuntimeError):eng.transition()
 parent['state']['best_map']['ndr_mean']=.1
 eng.initialize_render(parent);eng.evaluate()
 assert not eng.state['history'][-1]['eligible_for_selection']
