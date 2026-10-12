"""Single-GPU sequential queue; failed/paused jobs require review, never auto-retry."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from experiments.bottleneck_fc.config import process_identity,process_identity_matches
from experiments.regularization.config import atomic_write_json
from experiments.robustness_suite.controller import gpu_ready,cuda_busy
from experiments.full_dataset_suite.config import ROOT,DEFAULT_OUT,ARMS,prepare
from experiments.full_dataset_suite.storage import require_disk_reserve

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def utc():return datetime.now(timezone.utc).isoformat()

def authorized_jobs(plan):
    if plan["arms"]!=list(ARMS):raise ValueError("queue must contain exactly authorized arms1-3 in order")
    return plan["arms"]

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=DEFAULT_OUT);args=parser.parse_args()
    root=args.root.resolve();plan=read(root/"queue_plan.json");authorized_jobs(plan)
    lock=root/"controller_lock.json"
    if lock.exists():raise RuntimeError("controller lock exists; inspect rather than auto-restart")
    own={"pid":os.getpid(),"identity":process_identity(os.getpid())}
    with lock.open("x",encoding="utf-8") as stream:json.dump(own,stream)
    registry={"state":"running","controller":own,"started_at":utc(),"plan":plan,
              "jobs":[{"arm":arm,"state":"prepared","out":str(root/"runs"/arm)} for arm in ARMS]}
    def write():
        registry["updated_at"]=utc();atomic_write_json(root/"queue_state.json",registry)
    write()
    try:
        for job in registry["jobs"]:
            output=Path(job["out"])
            if (output/"last.pt").exists():raise RuntimeError("existing checkpoint must be reviewed; automatic restart forbidden")
            while True:
                busy=cuda_busy((os.getpid(),));ready,gpu=gpu_ready()
                for proc in psutil.process_iter(["pid","name","cmdline"]):
                    try:
                        cmd=" ".join(proc.info["cmdline"] or [])
                        if proc.pid!=os.getpid() and "python" in (proc.info["name"] or "").lower() and "experiments.full_dataset_suite.run" in cmd:
                            busy.append({"pid":proc.pid,"command":cmd})
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                if not busy and ready:break
                registry.update(state="waiting_for_gpu",gpu=gpu,busy=busy);write();time.sleep(10)
            require_disk_reserve(root);execution=prepare(output,job["arm"])
            smoke_path=root/"smoke_verified"/job["arm"]/"smoke_verification.json"
            if not smoke_path.exists():raise RuntimeError("arm GPU smoke not verified")
            smoke=read(smoke_path)
            if not smoke.get("passed") or smoke.get("source_input_identity")!=execution["file_identities"] or smoke.get("policy")!=execution["policy"]:
                raise RuntimeError("smoke is not valid for current immutable source/input/policy")
            command=[sys.executable,"-u","-m","experiments.full_dataset_suite.run","run","--arm",job["arm"],"--out",str(output)]
            with (output/"train_stdout.log").open("a",encoding="utf-8") as stdout,(output/"train_stderr.log").open("a",encoding="utf-8") as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            job.update(state="running",pid=child.pid,identity=process_identity(child.pid),command=command)
            registry["state"]="running";write()
            while child.poll() is None:
                status=output/"status.json"
                if status.exists():
                    try:job["latest_status"]=read(status)
                    except (OSError,json.JSONDecodeError):pass
                write();time.sleep(5)
            job["exit_code"]=child.returncode
            job["latest_status"]=read(output/"status.json") if (output/"status.json").exists() else {}
            if child.returncode==0 and job["latest_status"].get("status")=="complete" and (output/"complete.json").exists():
                job["state"]="complete";job["result"]=read(output/"complete.json");write()
            else:
                job["state"]=job["latest_status"].get("status","failed")
                registry["state"]="attention_required";write();return
        registry["state"]="complete";write()
    except BaseException as error:
        registry.update(state="controller_failed",error=repr(error));write();raise
    finally:
        if lock.exists() and read(lock)==own:lock.unlink()

if __name__=="__main__":main()
