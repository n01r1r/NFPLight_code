import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tokenize

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colab_tools import attach_drive_dataset, monitor, relative_path, relocate_manifest, TrainingLog, plot_training, validation_loss


@pytest.mark.parametrize('name,profile', [('nfplight', 'nfplight_32_noquant'), ('unetpp', 'nested_32_noquant')])
def test_notebook_has_only_setup_data_training_live_cells(name, profile):
    notebook = nbformat.read(ROOT / f'notebooks/{name}.ipynb', as_version = 4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 8
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            assert cell.source.startswith('## ') and len(cell.source.splitlines()) == 1
        else:
            compile(cell.source, '<cell>', 'exec')
            assert not cell.outputs and cell.execution_count is None
            assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline))
    text = json.dumps(notebook)
    assert '%%writefile' not in text and 'iframe' not in text and 'offline_report.html' not in text
    assert profile in text and 'drive.mount' in text
    assert 'inspect.getsource' not in text and 'plot_pipeline' not in text and 'PROFILES' not in text
    source = (ROOT / 'colab_tools.py').read_text(encoding = 'utf-8')
    assert 'clear_output(wait = True)' in source and 'from tqdm import tqdm' in source
    assert 'import serve' not in source and 'iframe' not in source


def test_profiles_change_only_architecture_and_name_and_factory_routes(monkeypatch):
    from nfplight import checkpoint
    import nfplight.nested_model
    baseline = checkpoint.read_config(ROOT / 'configs/nfplight_32_noquant.json')
    nested = checkpoint.read_config(ROOT / 'configs/nested_32_noquant.json')
    comparable = copy.deepcopy(nested)
    assert comparable['network'].pop('architecture') == 'nested'
    comparable['profile'] = baseline['profile']
    assert comparable == baseline
    assert baseline['features']['synthetic_quantization'] == 'none'
    monkeypatch.setattr(checkpoint, 'NFPLight', lambda **kwargs: ('baseline', kwargs))
    monkeypatch.setattr(nfplight.nested_model, 'NestedModel', lambda **kwargs: ('nested', kwargs))
    assert checkpoint.build_model(baseline)[0] == 'baseline'
    assert checkpoint.build_model(nested)[0] == 'nested'
    bad = copy.deepcopy(nested)
    bad['network']['architecture'] = 'unknown'
    with pytest.raises(ValueError, match = 'unknown network'):
        checkpoint.build_model(bad)


@pytest.mark.parametrize('path', ['/content/data', 'C:\\data', '../data', 'a/../../data'])
def test_relative_path_guard(path):
    with pytest.raises(ValueError):
        relative_path(path)


def data_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path('drive/MyDrive/MatSynth_PRIMA_Fabric')
    rows = []
    for split, category, material in [('train', 'Fabric', 'a'), ('val', 'Leather', 'b')]:
        path = root / 'source_shards/train' / category / (material + '.npz')
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b'metadata fixture only')
        rows.append({'split':split, 'category':category, 'material_id':material, 'group_id':material,
                     'shard_path':f'C:\\old\\source_shards\\train\\{category}\\{material}.npz'})
    rows.append({'split':'test', 'shard_path':'sealed-test-must-not-be-read.npz'})
    manifest = root / 'manifests/fabric_leather_training_manifest.jsonl'
    manifest.parent.mkdir()
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return root, manifest, rows


def test_drive_attachment_never_replaces_other_data(tmp_path, monkeypatch):
    root, manifest, rows = data_fixture(tmp_path, monkeypatch)
    assert attach_drive_dataset(root, root) == root
    Path('other').mkdir()
    with pytest.raises(FileExistsError):
        attach_drive_dataset(root, 'other')
    observed = []
    monkeypatch.setattr(Path, 'symlink_to', lambda self, target, **kwargs: observed.append((self, target, kwargs)))
    attach_drive_dataset(root, 'data/MatSynth_PRIMA_Fabric')
    assert observed == [(Path('data/MatSynth_PRIMA_Fabric'), root.resolve(), {'target_is_directory':True})]
    assert manifest.is_file() and len(rows) == 3


def test_manifest_population_order_identity_and_guards(tmp_path, monkeypatch):
    root, manifest, rows = data_fixture(tmp_path, monkeypatch)
    original = manifest.read_bytes()
    counts = {'train/Fabric':1, 'val/Leather':1}
    summary = relocate_manifest(manifest, root, 'work/portable.jsonl', expected_counts = counts)
    output = Path('work/portable.jsonl')
    portable = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['material_id'] for row in portable] == ['a', 'b']
    assert all(not Path(row['shard_path']).is_absolute() for row in portable)
    assert summary['official_test_payloads_read'] == 0 and summary['materials'] == counts
    timestamp = output.stat().st_mtime_ns
    relocate_manifest(manifest, root, output, expected_counts = counts)
    assert output.stat().st_mtime_ns == timestamp and manifest.read_bytes() == original
    rows[1]['group_id'] = 'a'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match = 'overlap'):
        relocate_manifest(manifest, root, 'work/bad.jsonl')
    rows[1]['group_id'] = 'b'
    rows[0]['shard_path'] = 'source_shards/../outside.npz'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match = 'traversal'):
        relocate_manifest(manifest, root, 'work/bad.jsonl')


def report_fixture(tmp_path, monkeypatch, model = 'nested'):
    import psutil
    monkeypatch.chdir(tmp_path)
    root = Path('runs/fixture')
    (root / 'run').mkdir(parents = True)
    profile = json.loads((ROOT / ('configs/nested_32_noquant.json' if model == 'nested' else 'configs/nfplight_32_noquant.json')).read_text())
    (root / 'launch.json').write_text(json.dumps({'profile':profile}))
    (root / 'process.json').write_text(json.dumps({'pid':os.getpid(), 'create_time':psutil.Process().create_time()}))
    return root


def append_row(root, step, *, stage = 'map', total = None, value = .4):
    row = {'stage':stage, 'step':step, 'total_step':step if total is None else total,
           'loss':value, 'ndrs_sum':value, 'normal_loss':value/4, 'diffuse_loss':value/4,
           'roughness_loss':value/4, 'specular_loss':value/4, 'render_loss':0.,
           'lr':.0005, 'gradient_norm':1., 'step_seconds':.1}
    with (root / 'run/train.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def test_native_monitor_final_update_partial_row_and_stage_separation(tmp_path, monkeypatch):
    root = report_fixture(tmp_path, monkeypatch)
    append_row(root, 1)
    reader = TrainingLog(root)
    initial = reader.snapshot()
    assert initial['profile']['network']['architecture'] == 'nested'
    with (root / 'run/train.jsonl').open('ab') as stream:
        stream.write(b'{"incomplete"')
    assert reader.snapshot()['latest']['total_step'] == 1
    content = (root / 'run/train.jsonl').read_bytes().splitlines()[0] + b'\n'
    (root / 'run/train.jsonl').write_bytes(content)
    append_row(root, 1, stage = 'render', total = 2, value = .8)
    snapshot = reader.snapshot()
    assert snapshot['train']['map:loss'][0][2] == .4
    assert snapshot['train']['render:loss'][0][2] == .8
    class Completed:
        returncode = 0
        def poll(self):
            append_row(root, 2, stage = 'render', total = 3)
            return 0
    result = monitor(root, Completed(), refresh_seconds = .01, plots = False)
    assert result['latest']['total_step'] == 3
    result = monitor(root, refresh_seconds = .01, plots = True, follow = False)
    assert result['latest']['total_step'] == 3
    assert result['profile'] == initial['profile']
    assert not list(root.rglob('*.html'))


def test_invalid_metrics_are_reported_and_model_not_constructed(tmp_path, monkeypatch):
    import torch
    root = report_fixture(tmp_path, monkeypatch)
    append_row(root, 1, value = float('nan'))
    with pytest.raises(ValueError, match = 'invalid training metric'):
        TrainingLog(root).snapshot()
    assert not torch.cuda.is_initialized()


def test_cuda_admission_rejects_before_model_or_output_creation(tmp_path, monkeypatch):
    import train_colab
    import torch
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(train_colab.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(train_colab.cli, 'build_model', lambda *args: pytest.fail('model must not be constructed'))
    for profile in ('nfplight_32_noquant', 'nested_32_noquant'):
        with pytest.raises(ValueError, match = 'select a CUDA GPU runtime'):
            train_colab.main(['--config', f'configs/{profile}.json', '--manifest', 'work/not_read.jsonl',
                              '--run-root', 'runs/admission_test_never_created'])
    assert not (ROOT / 'runs/admission_test_never_created').exists()
    assert not torch.cuda.is_initialized()


def test_launcher_publishes_profile_before_process_and_resume_is_strict():
    tree = ast.parse((ROOT / 'train_colab.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'write_json']
    locations = {}
    for node in calls:
        path = ast.unparse(node.args[0])
        for name in ('launch.json', 'process.json'):
            if name in path:
                locations[name] = node.lineno
    assert locations['launch.json'] < locations['process.json']
    from nfplight.cli import validate_resume
    profile = json.loads((ROOT / 'configs/nested_32_noquant.json').read_text())
    runtime = json.loads((ROOT / 'configs/runtime_accum4.json').read_text())
    changed = copy.deepcopy(profile)
    changed['network']['dim'] = 16
    with pytest.raises(ValueError, match = 'architecture conflict'):
        validate_resume({'clean_profile':profile}, {}, changed, runtime, device = 'cuda')


def test_loss_plot_is_one_by_two_with_log_axes_and_single_update_visible():
    import matplotlib.pyplot as plt
    import torch
    row = {'stage':'render', 'total_step':1001, 'populations':{'heldout':{'mean':{'ndrs_sum':.2, 'render_l1':.1}}}}
    assert validation_loss(row, 2.) == .4
    snapshot = {'train':{'map:loss':[[1,.5,.5]]}, 'validation':[row], 'render_weight':2.}
    figure = plot_training(snapshot)
    assert len(figure.axes) == 2
    assert all(axis.get_yscale() == 'log' for axis in figure.axes)
    assert figure.axes[0].lines[0].get_marker() == '.'
    plt.close(figure)
    assert not torch.cuda.is_initialized()


def test_last_and_selected_best_are_preserved_without_replacing_an_unimproved_best(tmp_path):
    import torch
    from train_colab import save_cpu_checkpoint
    root, backup = tmp_path / 'run', tmp_path / 'drive'
    backup.mkdir()
    selected = {'stage':'map', 'step':1, 'total_step':1, 'metric':.5,
                'net_g':{'fixture':torch.tensor([1.], dtype=torch.float32)}}
    payload = {'state':{'total_step':1, 'best_map':selected, 'best_render':None},
               'clean_profile':{}, 'identity':{}, 'relation_version':'fixture', 'optimizer':{}, 'rng':{}}
    save_cpu_checkpoint(root / 'last.pt', payload, reserve_bytes=0, backup_dir=backup)
    assert torch.load(backup / 'last.pt', weights_only=True)['state']['total_step'] == 1
    best = torch.load(backup / 'best.pt', weights_only=True)
    assert best['selected']['metric'] == .5 and 'optimizer' not in best and 'rng' not in best
    timestamp = (backup / 'best.pt').stat().st_mtime_ns
    payload['state']['total_step'] = 2
    save_cpu_checkpoint(root / 'last.pt', payload, reserve_bytes=0, backup_dir=backup)
    assert (backup / 'best.pt').stat().st_mtime_ns == timestamp
    assert torch.load(backup / 'last.pt', weights_only=True)['state']['total_step'] == 2
