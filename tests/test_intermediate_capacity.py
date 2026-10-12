from pathlib import Path
import copy
import json
import pytest
from experiments.intermediate_capacity import run as screen
from experiments.regularization import run as trainer


def test_intermediate_plan_changes_only_capacity_and_selects_one_seed(tmp_path, monkeypatch):
    parent=tmp_path/'parent'
    parent.mkdir()
    config=json.loads((screen.ROOT/'experiments/regularization/config.json').read_text(encoding='utf-8'))
    source=parent/'source.py'
    source.write_text('original',encoding='utf-8')
    execution=dict(experiment_settings=config,resolved_inputs=dict(manifest='unused',train=[],unseen_crop=[],heldout=[]),file_identities=[screen.file_identity(source)])
    (parent/'execution_config.json').write_text(json.dumps(execution),encoding='utf-8')
    control=parent/'seed19_d4_only'
    control.mkdir()
    (control/'complete.json').write_text(json.dumps(dict(seed=19,condition='d4_only')),encoding='utf-8')
    monkeypatch.setattr(screen,'PARENT',parent)
    out=tmp_path/'new_run'
    plan=screen.prepare(out)
    assert plan['seed']==19 and plan['settings']['training']['seeds']==[19]
    assert plan['settings']['model']['parameters']==15085034
    assert plan['settings']['model']['backbone']['middle_blocks']==12
    original=trainer.effective_settings(execution)
    assert plan['settings']['dataset']==original['dataset']
    assert plan['settings']['sampling']==original['sampling']
    assert plan['settings']['rendering']==original['rendering']
    for key in ('map_updates','render_updates','map_learning_rate','render_learning_rate','consecutive_intervals','relative_improvement_threshold'):
        assert plan['settings']['training'][key]==original['training'][key]
    assert plan['condition']['d4'] and not plan['condition']['dropout']
    assert screen.prepare(out)==plan
    source.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='source/data/control changed'):
        screen.verify_plan(plan)


def test_transition_refuses_failed_guard_before_gpu_work(tmp_path, monkeypatch):
    parent=tmp_path/'parent'
    parent.mkdir()
    out=tmp_path/'new_run'
    out.mkdir()
    (parent/'seed19_guard_status.json').write_text(json.dumps(dict(status='failed',error='test boundary failure')),encoding='utf-8')
    monkeypatch.setattr(screen,'PARENT',parent)
    with pytest.raises(RuntimeError,match='guard failed'):
        screen.wait_parent(out)
    assert screen.read(out/'transition_status.json')['status']=='waiting_for_seed19_conclusion'


@pytest.mark.parametrize('identity', ['live-creation', None])
def test_resume_rejects_live_or_unverified_process_before_runtime(tmp_path, monkeypatch, identity):
    from experiments.bottleneck_fc import config
    parent = tmp_path / 'parent'; parent.mkdir()
    (parent / 'two_seed_conclusion.json').write_text('{}')
    out = tmp_path / 'run'; out.mkdir()
    (out / 'status.json').write_text(json.dumps({'status': 'running', 'pid': 123, 'process_identity': identity}))
    monkeypatch.setattr(screen, 'PARENT', parent)
    monkeypatch.setattr(screen, 'verify_plan', lambda plan: None)
    monkeypatch.setattr(config, 'process_identity', lambda pid: 'live-creation')
    monkeypatch.setattr(trainer, '_runtime_environment', lambda settings: pytest.fail('runtime reached'))
    with pytest.raises(RuntimeError, match='active or unverified'):
        screen.run(out, {}, resume=True)
    assert screen.read(out / 'status.json')['status'] == 'running'


def test_export_destination_exists_before_evaluator(tmp_path):
    from experiments.bottleneck_fc.run import _atomic_npz
    import numpy as np
    directory = tmp_path / 'seed19_d4_only' / 'arrays'
    _atomic_npz(directory / 'sample.npz', prediction=np.ones(1, dtype=np.float32))
    with np.load(directory / 'sample.npz') as data:
        assert data['prediction'].dtype == np.float32
