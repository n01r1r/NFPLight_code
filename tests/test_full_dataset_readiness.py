"""Readiness failures must be detected before CUDA or production output."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.full_dataset_suite import config, run
from model.relation_geometry import RELATION_VERSION


def test_inference_export_preserves_feature_geometry_and_backbone(tmp_path):
    from experiments.full_dataset_suite.reporting import inference_payload
    from model.nfplight_model import NFPLightModel, checkpoint_feature_version
    execution = {"scientific_settings": {
        "rendering": {"feature_version": "raw_calibrated_gain9_v3", "relation_version": RELATION_VERSION},
        "backbone": {"width": 32, "middle_blocks": 4}}}
    selected = {"net_g": {"weight": torch.ones(1)}}
    payload = inference_payload(selected, execution)
    path = tmp_path / "export.pth"
    torch.save(payload, path)
    restored = torch.load(path, weights_only=True)
    model = NFPLightModel.__new__(NFPLightModel)
    model.feature_version = checkpoint_feature_version(restored)
    model.init_distance()
    assert (model.near_distance, model.far_distance) == (4., 12.)
    assert restored["relation_version"] == RELATION_VERSION
    assert restored["normal_decoder"] == "unit XYZ"
    assert restored["precision"] == "float32"
    assert torch.equal(restored["net_g"]["weight"], selected["net_g"]["weight"])
    payload["backbone"]["middle_blocks"] = 99
    assert execution["scientific_settings"]["backbone"]["middle_blocks"] == 4
    with pytest.raises(KeyError):
        inference_payload(selected, {"scientific_settings": {"rendering": {}}})


def shard(tmp_path, *, shape=(16,256,256,3), dtype=np.uint8):
    path=tmp_path/'material.npz'
    rgb=np.zeros(shape,dtype=dtype)
    np.savez(path,normal=rgb,diffuse=rgb,roughness=np.zeros((16,256,256),np.uint8),specular=rgb)
    return {'split':'train','shard_path':str(path)}


def test_all_required_bank_headers_are_checked(tmp_path):
    row=shard(tmp_path)
    result=config.inspect_shard_headers([row])
    assert result['passed'] and result['map_headers']==4
    assert result['official_test_payloads_read']==0 and not result['sha256_computed']


@pytest.mark.parametrize('shape,dtype', [((15,256,256,3),np.uint8),((16,256,256,3),np.float32)])
def test_invalid_bank_shape_or_dtype_rejected(tmp_path,shape,dtype):
    with pytest.raises(ValueError,match='invalid bank map'):
        config.inspect_shard_headers([shard(tmp_path,shape=shape,dtype=dtype)])


def test_official_test_payload_never_opened(tmp_path):
    with pytest.raises(ValueError,match='official test'):
        config.inspect_shard_headers([{'split':'test','shard_path':str(tmp_path/'does-not-exist.npz')}])


def test_duplicate_payload_and_wrong_category_population_rejected(tmp_path):
    rows=[json.loads(line) for line in config.MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=[r for r in rows if r['split'] in {'train','val'}]
    selected[1]['shard_path']=selected[0]['shard_path']
    path=tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in selected),encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate shard'):
        config.dataset_rows(path)
    selected[1]['shard_path']='different.npz'
    selected[0]['category']='Unknown'
    path.write_text('\n'.join(json.dumps(r) for r in selected),encoding='utf-8')
    with pytest.raises(ValueError,match='category populations'):
        config.dataset_rows(path)


def test_prepare_is_idempotent_and_smoke_cannot_be_reused_for_training(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(config,'execution_identity',lambda arm,purpose: {'purpose':purpose,'train':[],'validation':[],'arm':arm})
    out=tmp_path/'artifacts/smoke'
    first=config.prepare(out,'baseline7m_p01',purpose='smoke')
    assert first==config.prepare(out,'baseline7m_p01',purpose='smoke')
    before=(out/'execution_config.json').read_bytes()
    with pytest.raises(ValueError,match='immutable'):
        config.prepare(out,'baseline7m_p01')
    assert (out/'execution_config.json').read_bytes()==before
    assert not list(out.glob('*.pt'))


@pytest.mark.parametrize('rendering,purpose,smoke', [({},'training',False),({'relation_version':RELATION_VERSION},'smoke',False)])
def test_pipeline_contract_rejected_before_model_or_cuda(monkeypatch,rendering,purpose,smoke):
    def forbidden(*args,**kwargs):
        pytest.fail('model construction must not occur')
    monkeypatch.setattr(run,'Model',forbidden)
    with pytest.raises(ValueError):
        run.Pipeline({'purpose':purpose,'scientific_settings':{'rendering':rendering}},smoke=smoke)


@pytest.mark.parametrize('status', ['memory_paused','disk_paused','failed','budget_paused'])
def test_paused_smoke_is_never_passed(status):
    with pytest.raises(RuntimeError,match='smoke incomplete'):
        run.require_smoke_updates({'status':status,'map_updates':2,'render_updates':2,'total_step':4},map_updates=2,render_updates=2)


def test_smoke_checks_counts_and_entire_restored_state():
    state={'status':'interrupted','map_updates':2,'render_updates':2,'total_step':4}
    run.require_smoke_updates(state,map_updates=2,render_updates=2)
    with pytest.raises(RuntimeError):
        run.require_smoke_updates(state,map_updates=2,render_updates=0)
    snapshot={'optimizer':{'exp_avg':torch.tensor([1.])},'rng':[torch.tensor([2]),(3,4)]}
    run.verify_smoke_restore(snapshot,copy.deepcopy(snapshot))
    corrupt=copy.deepcopy(snapshot);corrupt['optimizer']['exp_avg'][0]=0
    with pytest.raises(AssertionError,match='optimizer'):
        run.verify_smoke_restore(snapshot,corrupt)
