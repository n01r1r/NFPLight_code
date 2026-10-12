"""Authorized render-only factor .8 continuation from frozen 15M24 map best."""
import argparse,copy,gc,os,shutil,traceback
from pathlib import Path
import psutil,torch
from experiments.regularization.config import atomic_write_json,file_identity
from experiments.robustness_suite import engine as base
from experiments.robustness_suite.run import Pipeline,process_lock,runtime_environment,record_runtime,memory_observation,memory_is_low
from experiments.robustness_suite.controller import cuda_busy,gpu_ready,utc
from experiments.capacity_crop_followup import guarded_save,require_space,DiskReserveError
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/"artifacts/capacity15m_extra24_seed19_20261010/run"
OUT=ROOT/"artifacts/render_factor08_15m24_seed19_20261010"
ARM="render_factor08_15m24_seed19"
def read(p):
 import json
 return json.loads(Path(p).read_text(encoding="utf-8"))
def identity():
 old=read(PARENT/"execution_config.json")
 if read(PARENT/"status.json")["status"]!="complete":raise ValueError("parent must be complete")
 for x in old["file_identities"]:
  s=Path(x["path"]).stat()
  if (s.st_size,s.st_mtime_ns)!=(x["size"],x["mtime_ns"]):raise ValueError("frozen parent dependency changed")
 new=copy.deepcopy(old);new["arm"]=ARM
 new["policy"].update(arm=ARM,factor=.8,render_cap=25000)
 new["origin"]={"parent":str(PARENT),"model":"best_map.pt: map500","phase_rng":"map_stage_end.pt: map7100","optimizer":"fresh Adam","scientific_change":"render factor .5 -> .8 only","budget_change":"render cap15000 ->25000; cap is pause, not convergence","primary_common_updates":7400,"historical_control":"parent render .5; causal claim requires matched trajectory evidence"}
 new["file_identities"] += [file_identity(p) for p in (Path(__file__),PARENT/"execution_config.json",PARENT/"best_map.pt",PARENT/"map_stage_end.pt")]
 new["file_identities"].sort(key=lambda x:x["path"])
 return new

def prepare(out):
 out=Path(out).resolve();candidate=identity()
 if out==PARENT or PARENT in out.parents:raise ValueError("new lineage required")
 path=out/"execution_config.json"
 if path.exists():
  if read(path)!=candidate:raise ValueError("immutable identity mismatch")
 else:
  if out.exists() and any(out.iterdir()):raise FileExistsError("new output must be empty")
  out.mkdir(parents=True,exist_ok=True)
  for x in candidate["file_identities"]:
   src=Path(x["path"])
   if src.suffix==".py" and src.is_relative_to(ROOT):
    dest=out/"source_snapshot"/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  atomic_write_json(path,candidate)
  atomic_write_json(out/"budget_ledger.json",[{"at_utc":utc(),"event":"initial_user_authorized_extra_budget","map_cap":candidate["policy"]["map_cap"],"render_cap":25000,"common_updates":7400}])
 return candidate

class RenderEngine(base.Engine):
 def initialize_render(self,parent):
  self.net.load_state_dict(parent["state"]["best_map"]["net_g"])
  self.state=copy.deepcopy(parent["state"])
  self.state.update(stage="render",stage_step=0,render_updates=0,status="running",selected=None)
  self.state["history"]=[];self.state["evaluations"]={"map":0,"render":0}
  self.state["best_render"]=None;self.state["convergence"]["render"]=self._new_convergence()
  self.state["render_start_sample_offset"]=self.state["map_updates"]
  self.state["origin_map_history_in_parent"]=True
  self.samples.setstate(parent["sample_rng"]);self.d4.setstate(parent["d4_rng"])
  self.augmentation_rng.setstate(parent["augmentation_rng"]);self.noise_generator.set_state(parent["noise_rng"])
  self._new_optimizer("render");base.restore_rng(parent["rng"])
  assert not self.optimizer.state
 def transition(self):raise RuntimeError("render-only lineage cannot train map")

def gate(out):
 storage=require_space(out,launch=True)
 busy=cuda_busy((os.getpid(),))
 for p in psutil.process_iter(["name","cmdline"]):
  try:
   args=p.info["cmdline"] or []
   if p.pid!=os.getpid() and "python" in (p.info["name"] or "").lower() and any(args[i:i+3] in (["-m","experiments.render_factor_followup","run"],["-m","experiments.render_factor_followup","smoke"],["-m","experiments.capacity_crop_followup","run"]) for i in range(len(args)-2)):busy.append({"pid":p.pid,"command":args})
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 ready,gpu=gpu_ready();ram=psutil.virtual_memory().available
 report={"at_utc":utc(),"ready":ready and not busy and ram>=2*2**30,"gpu":gpu,"busy":busy,"ram":ram,"storage":storage}
 atomic_write_json(out/"launch_gate.json",report)
 if not report["ready"]:raise RuntimeError("GPU/RAM launch gate not ready")

def run(out,resume=False,smoke=False):
 out=Path(out).resolve();ex=prepare(out)
 if not resume and (out/"last.pt").exists():raise FileExistsError("explicit resume required")
 gate(out)
 with process_lock(out,resume=resume) as process:
  environment=runtime_environment()
  if (out/"runtime.json").exists() and read(out/"runtime.json")["environment"]!=environment:raise ValueError("resume environment changed")
  record_runtime(out,environment,process)
  pipe=Pipeline(ex)
  eng=RenderEngine(out,pipe.model.net_g,ex["policy"],ex,pipe.objective,pipe.evaluate,sample_count=len(pipe.raw_train),adam_betas=tuple(pipe.settings["training"]["adam_betas"]),clip_norm=pipe.settings["training"]["gradient_clip_norm"])
  pipe.engine=eng;eng.memory_guard=lambda:memory_is_low(memory_observation())
  original=eng._write_status
  def write_status(status=None,error=None):
   original(status,error);doc=read(out/"status.json");doc.update(process,memory=memory_observation(),output_directory=str(out));atomic_write_json(out/"status.json",doc)
  eng._write_status=write_status
  base.atomic_save=guarded_save
  try:
   if not resume:
    parent=base.load_checkpoint(PARENT/"map_stage_end.pt");eng.initialize_render(parent);del parent;gc.collect()
    eng.evaluate();z=eng.state["history"][-1];expected=next(x for x in read(PARENT/"evaluation.json") if x["stage"]=="render" and x["step"]==0)
    measured=z["populations"]["heldout"]["mean"];reference=expected["populations"]["heldout"]["mean"]
    if any(abs(measured[k]-reference[k])>1e-7 for k in reference):raise ValueError("canonical render0 does not reproduce parent evaluation")
    atomic_write_json(out/"initialization_verification.json",{"at_utc":utc(),"render0_matches_parent":True,"heldout_mean":measured,"fresh_adam":not eng.optimizer.state,"sample_offset":eng.state["render_start_sample_offset"],"factor":.8,"initial_lr":eng.optimizer.param_groups[0]["lr"],"cuda_rng_restored":True,"scope":"parent map-end RNG + selected best-map; no original render0 full checkpoint claim"})
    eng.committed_step=0;eng.commit();base.atomic_save(out/"render_start.pt",eng.snapshot())
   caps={"map_cap":ex["policy"]["map_cap"],"render_cap":1 if smoke else 25000}
   result=eng.run(caps,resume=True)
   if smoke:atomic_write_json(out/"smoke.json",{"at_utc":utc(),"render_updates":result["render_updates"],"status":result["status"],"initialization":read(out/"initialization_verification.json"),"not_real_training":True})
   return result
  except (torch.OutOfMemoryError,MemoryError) as error:
   traceback.clear_frames(error.__traceback__);eng.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
   last=out/"last.pt";eng.restore(base.load_checkpoint(last if last.exists() else out/"previous.pt"));eng.committed_step=eng.state["total_step"]
   from experiments.regularization.run import _truncate_log
   _truncate_log(out/"train.jsonl",eng.committed_step);write_status("memory_paused",error);return eng.state
  except DiskReserveError as error:
   write_status("storage_paused",error);raise

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","smoke","run"));p.add_argument("--out",type=Path);p.add_argument("--resume",action="store_true");a=p.parse_args()
 out=a.out or OUT/("smoke" if a.command=="smoke" else"run")
 if a.command=="prepare":prepare(out);print(str(out));return
 if a.resume and a.command!="run":p.error("resume is run-only")
 print(run(out,resume=a.resume,smoke=a.command=="smoke")["status"])
if __name__=="__main__":main()
