import json
import random
from types import SimpleNamespace

import pytest
import torch

import benchmark_fabric_leather as benchmark
import train_fabric_leather as trainer


def target():
    maps = torch.zeros(1, 10, 2, 2)
    maps[:, 2] = 1
    maps[:, 7:] = -.92
    return maps


def test_ndrs_sum_weights_maps_equally_and_supervises_specular():
    gt = torch.zeros(1, 10, 2, 2)
    pred = gt.clone()
    for section, value in [(slice(0, 3), .3), (slice(3, 6), .6),
                           (slice(6, 7), .9), (slice(7, 10), 1.2)]:
        pred[:, section] = value
    pred.requires_grad_()
    loss, parts = benchmark.ndrs_loss(pred, gt, torch.ones(1, 1, 2, 2, dtype=torch.bool))
    torch.testing.assert_close(loss, torch.tensor(3.))
    assert len(parts) == 4
    loss.backward()
    assert pred.grad[:, 7:].gt(0).all()


def test_sum_ignores_invalid_pixels_and_rejects_missing_support():
    pred = torch.ones(1, 10, 1, 2)
    pred[:, :, :, 1] = 100
    gt = torch.zeros_like(pred)
    mask = torch.tensor([[[[True, False]]]])
    loss, _ = benchmark.ndrs_loss(pred, gt, mask)
    torch.testing.assert_close(loss, torch.tensor(4.))
    with pytest.raises(ValueError, match='zero valid support'):
        benchmark.ndrs_loss(pred, gt, torch.zeros_like(mask))
    pred[:, 7, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='non-finite'):
        benchmark.ndrs_loss(pred, gt, mask)


def model():
    net = torch.nn.Conv2d(1, 10, 1)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
        net.bias[2] = 2
    return SimpleNamespace(net_g=net, device='cpu')


def prepare(_model, maps):
    return maps, torch.ones(maps.shape[0], 1, 2, 2, dtype=torch.bool), torch.ones(maps.shape[0], 1, 2, 2)


def test_real_update_logs_sum_and_trains_specular_with_unit_normal(monkeypatch):
    monkeypatch.setattr(benchmark, 'prepare', prepare)
    bank = SimpleNamespace(sample=lambda rng, count, fixed: (target(), []))
    net = model()
    optimizer = torch.optim.SGD(net.net_g.parameters(), lr=.01)
    before = net.net_g.bias[7:].detach().clone()
    row = benchmark.update(net, bank, optimizer, random.Random(0), None, batch=1, accumulate=1)
    assert row['map_aggregation'] == 'ndrs_sum'
    assert row['map_loss'] == pytest.approx(sum(row[k] for k in ['normal_loss', 'diffuse_loss', 'roughness_loss', 'specular_loss']))
    assert row['normal_loss'] == 0
    assert row['specular_loss'] == pytest.approx(.92)
    assert (net.net_g.bias[7:].detach() < before).all()
    unit = benchmark.unit_prediction(net.net_g(torch.ones(1, 1, 2, 2)))
    torch.testing.assert_close(unit[:, :3].norm(dim=1), torch.ones(1, 2, 2))


def test_full_evaluation_reports_four_terms_and_sum(monkeypatch):
    monkeypatch.setattr(trainer, 'prepare', prepare)
    monkeypatch.setattr(trainer, 'render_batch', lambda model, maps, lights, views: torch.zeros(1, 40, 3, 2, 2))
    bank = SimpleNamespace(records=[{'material_id': 'fabric', 'category': 'Fabric'},
                                   {'material_id': 'leather', 'category': 'Leather'}],
                           crop=lambda index, crop: target()[0])
    result = trainer.evaluate(model(), bank, torch.zeros(40, 3))
    assert len(result['materials']) == 2
    assert result['mean']['map_specular'] == pytest.approx(.92)
    assert result['mean']['ndrs_sum'] == pytest.approx(.92)
    assert result['mean']['ndr_mean'] == 0  # Historical diagnostic only.


def test_bank_keeps_all_training_materials(tmp_path):
    rows = [{'split': 'train', 'category': 'Fabric' if i < 5 else 'Leather', 'material_id': str(i)} for i in range(10)]
    rows.append({'split': 'val', 'category': 'Fabric', 'material_id': 'val'})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    assert len(benchmark.Bank(manifest, 'train').records) == 10
    assert len(benchmark.Bank(manifest, 'val').records) == 1


def test_new_contract_records_sum_fixed_specular_and_all_materials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = ['train_fabric_leather.py', 'benchmark_fabric_leather.py', 'train_fabric.py',
             'data/fabric.py', 'model/nfplight_model.py', 'network/nfplight_net.py', 'utils/render_util.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture', encoding='utf8')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}', encoding='utf8')
    shard = tmp_path / 'shard.npz'
    shard.write_bytes(b'fixture')
    train = SimpleNamespace(records=[{'material_id': str(i), 'category': 'Fabric', 'shard_path': str(shard)} for i in range(6)])
    val = SimpleNamespace(records=[{'material_id': 'val', 'category': 'Leather', 'shard_path': str(shard)}])
    args = SimpleNamespace(manifest=str(manifest), map_steps=10, render_steps=10, batch=4,
                           accumulate=1, val_every=100, checkpoint_every=100, seed=19,
                           validation_material_limit=0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU fixture')
    contract = trainer.contract_for(args, train, val)
    assert contract['format'] == 'nfplight.fabric_leather.contract.v2'
    assert contract['map_monitor'] == 'ndrs_sum'
    assert '/3' not in contract['map_reduction']
    assert contract['normal_loss'] == 'unit XYZ L1'
    assert contract['specular_supervision'] is True
    assert contract['specular_target'] == .04
    assert contract['f0_input_and_both_render_branches'] == .04
    assert contract['train_materials'] == 6
    assert len(contract['train_material_ids']) == 6
    assert contract['sha256_computed'] is False
