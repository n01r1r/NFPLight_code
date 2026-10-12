"""Serial single-GPU queue for the authorized robustness suite."""
from __future__ import annotations
import argparse,datetime,json,os,pathlib,subprocess,sys,time
import psutil
from experiments.bottleneck_fc.config import process_identity,process_identity_matches
from experiments.regularization.config import atomic_write_json

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def eligible_next(registry):
 for job in registry['jobs']:
  if job['state'] in ('failed','memory_paused','budget_paused','interrupted'):
   raise RuntimeError('stopped job requires review before continuing the queue')
  if job['state'] in ('planned','prepared','running'):return job
 return None

def terminal(status):
 return status in ('complete','failed','memory_paused','budget_paused','interrupted')

def cuda_busy(excluded=()):
 active=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  try:
   if p.pid in excluded or 'python' not in (p.info['name'] or '').lower():continue
   command=' '.join(p.info['cmdline'] or [])
   if 'live_training_plot' in command:continue
   if any(x in command for x in ('train_fabric','experiments.plateau_followup.run run','experiments.robustness_suite.run run','experiments.robustness_suite.run smoke','experiments.robustness_suite.diagnostics')):
    active.append({'pid':p.pid,'command':command,'identity':process_identity(p.pid)})
  except (psutil.NoSuchProcess,psutil.AccessDenied):continue
 return active

def gpu_ready():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 free,util=(float(x.strip()) for x in p.stdout.splitlines()[0].split(','))
 compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
 cuda_pids=[]
 for line in compute.stdout.splitlines():
  fields=line.split(',')
  if fields and fields[0].strip().isdigit():cuda_pids.append(int(fields[0].strip()))
 mode=subprocess.run(['nvidia-smi','--query-gpu=driver_model.current','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 # WDDM also lists desktop graphics clients. Keep them visible for review.
 # Python compute processes, trainer commands, memory and utilization guard launch.
 blocking=cuda_pids
 if 'WDDM' in mode.upper():
  blocking=[]
  for pid in cuda_pids:
   try:
    if 'python' in psutil.Process(pid).name().lower():blocking.append(pid)
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 return free>=14000 and util<=20 and not blocking,{'free_mib':free,'utilization_percent':util,'driver_model':mode,'reported_gpu_pids':cuda_pids,'blocking_compute_pids':blocking}

def write(root,registry):
 registry['updated_at']=utc();atomic_write_json(root/'queue_state.json',registry)

def initialize(root,plan):
 return {'version':1,'plan':plan,'state':'running','started_at':utc(),'controller':{'pid':os.getpid(),'identity':process_identity(os.getpid())},'jobs':[{'arm':a,'state':'prepared' if (root/'arms_final'/a/'execution_config.json').is_file() else 'planned','out':str((root/'arms_final'/a).resolve())} for a in plan['arms']], 'events':[]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args();root=pathlib.Path(args.root).resolve();plan=json.loads((root/'queue_plan.json').read_text(encoding='utf-8'));lock=root/'controller_lock.json'
 if lock.exists():
  old=json.loads(lock.read_text());
  if process_identity_matches(old['pid'],old['identity']):raise RuntimeError('controller already active')
  if not args.resume:raise RuntimeError('stale controller requires explicit --resume')
  lock.unlink()
 with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'identity':process_identity(os.getpid())},f)
 registry=json.loads((root/'queue_state.json').read_text()) if args.resume and (root/'queue_state.json').exists() else initialize(root,plan)
 if registry['plan']!=plan:raise ValueError('immutable queue plan changed')
 registry['controller']={'pid':os.getpid(),'identity':process_identity(os.getpid())};registry['state']='running';write(root,registry)
 try:
  while True:
   job=eligible_next(registry)
   if job is None:
    registry['state']='complete';write(root,registry);break
   out=pathlib.Path(job['out']);statuspath=out/'status.json';launchpath=out/'queue_launch.json';status=json.loads(statuspath.read_text()) if statuspath.exists() else {}
   if status.get('status')=='complete':job['state']='complete';job['complete']=json.loads((out/'complete.json').read_text());write(root,registry);continue
   prior=json.loads(launchpath.read_text()) if launchpath.exists() else None
   if prior and process_identity_matches(prior['pid'],prior['identity']):
    child=None;pid=prior['pid'];job['state']='running';write(root,registry)
   else:
    if job['state']=='running' or (out/'last.pt').exists():
     registry['state']='attention_required';job['state']='interrupted';registry['events'].append({'at':utc(),'event':'dead_child_requires_review','arm':job['arm'],'status':status});write(root,registry);break
    busy=cuda_busy((os.getpid(),));ready,gpu=gpu_ready()
    if busy or not ready:
     registry['state']='waiting_for_gpu';registry['gpu']=gpu;registry['gpu_processes']=busy;write(root,registry);time.sleep(10);continue
    time.sleep(5);ready,gpu=gpu_ready()
    if not ready:continue
    registry['state']='running'
    preparation=subprocess.run([sys.executable,'-m','experiments.robustness_suite.run','prepare','--arm',job['arm'],'--out',str(out)],capture_output=True,text=True,cwd=str(pathlib.Path(__file__).resolve().parents[2]))
    if preparation.returncode:
     job['state']='failed';registry['state']='attention_required';job['error']=preparation.stderr[-4000:];write(root,registry);break
    command=[sys.executable,'-u','-m','experiments.robustness_suite.run','run','--arm',job['arm'],'--out',str(out)]
    with (out/'train_stdout.log').open('a',encoding='utf-8') as stdout,(out/'train_stderr.log').open('a',encoding='utf-8') as stderr:
     child=subprocess.Popen(command,cwd=str(pathlib.Path(__file__).resolve().parents[2]),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    pid=child.pid;identity=process_identity(pid);atomic_write_json(launchpath,{'pid':pid,'identity':identity,'command':command,'started_at':utc()});job.update(state='running',pid=pid,identity=identity);write(root,registry)
   while process_identity_matches(pid,job.get('identity',prior['identity'] if prior else None)):
    if statuspath.exists():
     try:job['latest_status']=json.loads(statuspath.read_text());write(root,registry)
     except (OSError,json.JSONDecodeError):pass
    time.sleep(5)
   if child is not None:job['exit_code']=child.wait()
   status=json.loads(statuspath.read_text()) if statuspath.exists() else {};job['latest_status']=status
   if status.get('status')=='complete' and (child is None or job['exit_code']==0):
    job['state']='complete';job['complete']=json.loads((out/'complete.json').read_text())
   else:
    job['state']=status.get('status','failed');registry['state']='attention_required';registry['events'].append({'at':utc(),'event':'job_stopped','arm':job['arm'],'status':status});write(root,registry);break
   write(root,registry)
 except BaseException as error:
  registry['state']='controller_failed';registry['error']=repr(error);write(root,registry);raise
 finally:
  if lock.exists():
   old=json.loads(lock.read_text())
   if old['pid']==os.getpid():lock.unlink()

if __name__=='__main__':main()
