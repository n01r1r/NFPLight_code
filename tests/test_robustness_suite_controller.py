from experiments.robustness_suite.controller import eligible_next,terminal

def test_queue_preserves_order_and_skips_complete():
 jobs=[{'arm':'A','state':'complete'},{'arm':'B','state':'prepared'},{'arm':'C','state':'planned'}]
 assert eligible_next({'jobs':jobs}) is jobs[1]
 jobs[1]['state']='complete'; assert eligible_next({'jobs':jobs}) is jobs[2]
 jobs[2]['state']='complete';assert eligible_next({'jobs':jobs}) is None

def test_pauses_require_review_and_no_automatic_retry():
 for value in ['failed','memory_paused','budget_paused','interrupted']:assert terminal(value)
 assert not terminal('running')
 import pytest
 with pytest.raises(RuntimeError,match='review'):
  eligible_next({'jobs':[{'state':'memory_paused'},{'state':'planned'}]})


def test_new_user_authorized_contract_context():
 from doctor_training_policy import validate_context
 context={"schema_version":1,"policy_authority":False,"configuration_scope":"current_immutable","authorized_contract":"docs/ROBUSTNESS_SUITE.md","canonical_policy":"docs/TRAINING_SCHEDULER_POLICY.md"}
 assert validate_context(context)==[]
 context["authorized_contract"]="docs/unapproved.md"
 assert validate_context(context)

def test_wddm_desktop_processes_do_not_block_but_python_compute_does(monkeypatch):
 from experiments.robustness_suite import controller
 from types import SimpleNamespace
 names={12:'dwm.exe',13:'chrome.exe',14:'python.exe'}
 def run(command,**kwargs):
  field=next(x for x in command if x.startswith('--query-'))
  return SimpleNamespace(stdout={'--query-gpu=memory.free,utilization.gpu':'22000, 1', '--query-gpu=driver_model.current':'WDDM','--query-compute-apps=pid,process_name':'12, unknown\n13, chrome.exe'}[field])
 monkeypatch.setattr(controller.subprocess,'run',run)
 monkeypatch.setattr(controller.psutil,'Process',lambda pid:SimpleNamespace(name=lambda:names[pid]))
 assert controller.gpu_ready()[0]
 def python_run(command,**kwargs):
  if '--query-compute-apps=pid,process_name' in command:return SimpleNamespace(stdout='14, python.exe')
  return run(command,**kwargs)
 monkeypatch.setattr(controller.subprocess,'run',python_run)
 assert not controller.gpu_ready()[0]
