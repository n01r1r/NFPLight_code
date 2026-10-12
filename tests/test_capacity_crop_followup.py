import pytest
import torch
from experiments import capacity_crop_followup as followup
from experiments.robustness_suite import config
from experiments.robustness_suite.run import create_network

def test_same_capacity_initial_state(monkeypatch):
    monkeypatch.setitem(config.ARMS, followup.ARM, {"parameters": followup.PARAMETERS, "middle_blocks": 12, "extra_training_crops": True})
    old = config.policy_for_arm("capacity15m_p01")
    new = config.policy_for_arm(followup.ARM)
    assert {k for k in old if old[k] != new[k]} == {"arm", "extra_training_crops"}
    torch.manual_seed(19)
    backbone = {"width":32, "middle_blocks":12, "encoder_blocks":[2,2,4], "decoder_blocks":[2,2,2]}
    a = create_network(old, backbone)
    torch.manual_seed(19)
    b = create_network(new, backbone)
    assert sum(x.numel() for x in b.parameters()) == followup.PARAMETERS
    assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())

def test_reserve_boundaries_and_committed_preservation(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    path.write_bytes(b"committed")
    warning, critical = followup.storage_limits(tmp_path)
    followup.require_space(tmp_path, free_bytes=critical)
    followup.require_space(tmp_path, launch=True, free_bytes=warning)
    with pytest.raises(followup.DiskReserveError):
        followup.require_space(tmp_path, free_bytes=critical-1)
    def blocked(*args, **kwargs):
        raise followup.DiskReserveError("blocked")
    monkeypatch.setattr(followup, "require_space", blocked)
    monkeypatch.setattr(followup, "_original_save", lambda *a, **k: pytest.fail("serialization called"))
    with pytest.raises(followup.DiskReserveError):
        followup.guarded_save(path, {}, rotate=True)
    assert path.read_bytes() == b"committed"
    assert not list(tmp_path.glob("*.tmp"))


def test_launcher_is_not_trainer():
    assert not followup.is_followup_trainer({"name":"pwsh.exe","cmdline":["pwsh.exe","-Command","python -m experiments.capacity_crop_followup smoke"]})
    assert followup.is_followup_trainer({"name":"python.exe","cmdline":["python.exe","-m","experiments.capacity_crop_followup","run","--out","x"]})
    assert not followup.is_followup_trainer({"name":"python.exe","cmdline":["python.exe","-m","experiments.capacity_crop_followup","status"]})
