from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise RuntimeError(message)


PRELUDE = '''
import os, sys, subprocess, types, json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
if torch.cuda.is_available() or torch.cuda.is_initialized():
    raise RuntimeError('Verification must have no visible CUDA GPU')
probe=subprocess.run([sys.executable,'-c','import torch; raise SystemExit(1 if torch.cuda.is_available() else 0)'], capture_output=True)
if probe.returncode:
    raise RuntimeError('Child CUDA visibility isolation failed; no training fixture may run')
real_check_call=subprocess.check_call
pip_checks=[]
def dependency_dry_run(command, *args, **kwargs):
    if command[1:4] == ['-m','pip','install']:
        result=subprocess.run([*command,'--dry-run','--no-index'],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        pip_checks.append({'dry_run':True,'network':False,'returncode':result.returncode})
        return 0
    return real_check_call(command,*args,**kwargs)
subprocess.check_call=dependency_dry_run
google=types.ModuleType('google')
colab=types.ModuleType('google.colab')
mount_calls=[]
colab.drive=types.SimpleNamespace(mount=lambda path:mount_calls.append(path))
google.colab=colab
sys.modules['google']=google
sys.modules['google.colab']=colab
'''


MONITOR = '''
import psutil
from colab_tools import monitor
fixture_root=Path('runs/native_plot_fixture')
(fixture_root/'run').mkdir(parents=True)
(fixture_root/'launch.json').write_text(json.dumps({'profile':profile}))
(fixture_root/'process.json').write_text(json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time()}))
def row(step):
    with (fixture_root/'run/train.jsonl').open('a') as stream:
        stream.write(json.dumps({'stage':'map','step':step,'total_step':step,'loss':.4/step,'ndrs_sum':.4/step,
            'normal_loss':.1/step,'diffuse_loss':.1/step,'roughness_loss':.1/step,'specular_loss':.1/step,
            'render_loss':0.,'lr':.0005,'gradient_norm':1.,'step_seconds':.1})+'\\n')
row(1)
class Completed:
    def poll(self):
        row(2)
        return 0
result=monitor(fixture_root,Completed(),refresh_seconds=.01,plots=False)
if result['latest']['total_step'] != 2:
    raise RuntimeError('Final child update lost')
result=monitor(fixture_root,refresh_seconds=.01,plots=True)
if result['profile'] != profile or not (fixture_root/'reports/training.png').is_file():
    raise RuntimeError('Native report profile/PNG mismatch')
if torch.cuda.is_initialized():
    raise RuntimeError('Unexpected CUDA initialization')
'''


def data_fixture(project):
    root = project / 'drive/MyDrive/MatSynth_PRIMA_Fabric'
    rows = []
    for split, category, count in [('train', 'Fabric', 358), ('train', 'Leather', 208),
                                    ('val', 'Fabric', 98), ('val', 'Leather', 66)]:
        for index in range(count):
            material = f'{split}_{category}_{index:04d}'
            shard = root / 'source_shards/train' / category / (material + '.npz')
            shard.parent.mkdir(parents = True, exist_ok = True)
            shard.touch()
            rows.append({'split':split, 'category':category, 'material_id':material, 'group_id':material,
                'shard_path':f'C:\\old\\source_shards\\train\\{category}\\{material}.npz'})
    rows.append({'split':'test', 'shard_path':'sealed-test-must-not-be-read.npz'})
    manifest = root / 'manifests/fabric_leather_training_manifest.jsonl'
    manifest.parent.mkdir()
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def git(project, *args):
    result = subprocess.run(['git', '-C', str(project), *args], capture_output = True, text = True)
    require(result.returncode == 0, result.stderr)


def main():
    delivered = nbformat.read(ROOT / 'notebooks/train.ipynb', as_version = 4)
    train_cell = next(cell.source for cell in delivered.cells if cell.cell_type == 'code'
                      and cell.source.startswith('from colab_tools import monitor\nif RUN_TRAINING:'))
    results = []
    work = ROOT / 'work'
    work.mkdir(exist_ok = True)
    with tempfile.TemporaryDirectory(prefix = 'notebook_check_', dir = work) as temporary:
        sandbox = Path(temporary).resolve()
        require(sandbox.is_relative_to(work.resolve()), 'verification directory outside workspace')
        for model in ('nfplight', 'naf_unetpp'):
            case = sandbox / model
            project = case / 'NFPLight_code'
            project.mkdir(parents = True)
            for folder in ('nfplight', 'configs'):
                shutil.copytree(ROOT / folder, project / folder, ignore = shutil.ignore_patterns('__pycache__'))
            for name in ('colab_tools.py', 'train_colab.py', 'preview_pipeline.py', 'requirements-colab.txt', '.gitignore'):
                shutil.copyfile(ROOT / name, project / name)
            git(project, 'init', '-b', 'codex/colab-experiments')
            git(project, 'remote', 'add', 'origin', 'https://github.com/n01r1r/NFPLight_code.git')
            git(project, 'add', '.')
            git(project, '-c', 'user.name=Notebook verification', '-c', 'user.email=verification@example.invalid',
                'commit', '-m', 'Isolated notebook fixture')
            data_fixture(project)
            notebook = nbformat.from_dict(delivered)
            notebook.cells.insert(0, nbformat.v4.new_code_cell(PRELUDE))
            index = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == 'code' and cell.source.startswith('from colab_tools import relative_path, attach_drive_dataset'))
            notebook.cells.insert(index, nbformat.v4.new_code_cell(f'''
MODEL={model!r}
MODEL_CONFIG=Path(PROFILES[MODEL])
RUN_NAME=MODEL+'_fresh_01'
RUN_ROOT=Path('runs')/RUN_NAME
DRIVE_MOUNT=PROJECT/'drive'
DATA_ROOT=Path('drive/MyDrive/MatSynth_PRIMA_Fabric')
SOURCE_MANIFEST=DATA_ROOT/'manifests/fabric_leather_training_manifest.jsonl'
'''))
            notebook.cells.append(nbformat.v4.new_code_cell(MONITOR))
            notebook.cells.append(nbformat.v4.new_code_cell('''
RUN_TRAINING=True
SHOW_PLOTS=False
RUN_NAME='no_cuda_admission'
RUN_ROOT=Path('runs')/RUN_NAME
if torch.cuda.is_available():
    raise RuntimeError('CUDA isolation lost before training admission fixture')
try:
    exec(compile(TRAIN_CELL,'<actual training cell>','exec'))
except RuntimeError as error:
    if 'select a CUDA GPU runtime' not in str(error):
        raise
else:
    raise RuntimeError('CUDA admission fixture should fail')
if RUN_ROOT.exists() or training_process.poll() == 0 or torch.cuda.is_initialized():
    raise RuntimeError('No-CUDA admission created model state/output')
Path('verification.json').write_text(json.dumps({'model':MODEL,'no_cuda_admission':True,
    'drive_mount_calls':mount_calls,'pip_dry_run':pip_checks,'model_execution':False}))
'''.replace('TRAIN_CELL', repr(train_cell))))
            client = NotebookClient(notebook, timeout = 120, kernel_name = 'python3',
                                    resources = {'metadata':{'path':str(case)}})
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'-1', 'IPYTHONDIR':str(case / '.ipython')}
            client.execute(env = env)
            outputs = [output for cell in notebook.cells for output in cell.get('outputs', [])]
            require(not any(output.output_type == 'error' for output in outputs), 'notebook emitted error')
            pngs = sum('image/png' in output.get('data', {}) for output in outputs)
            require(pngs >= 2, 'expected pipeline and native training PNG outputs')
            html = '\n'.join(output.get('data', {}).get('text/html', '') for output in outputs)
            require('<iframe' not in html.lower() and '<canvas' not in html.lower(), 'HTML plot leaked into notebook')
            result = json.loads((project / 'verification.json').read_text())
            result.update(delivered_cells=len(delivered.cells), inline_pngs=pngs)
            results.append(result)
    evidence = {'verified_at':datetime.now(timezone.utc).isoformat(), 'models':results,
        'actual_colab_frontend':False, 'actual_drive_mount':False, 'remote_clone':False,
        'drive_symlink_creation':'mocked API unit check; fixture reuses same existing source path on Windows',
        'dependencies':'installed versions resolved with pip --dry-run --no-index; no installation',
        'real_dataset_payloads_read':0, 'gpu_training_executed':False, 'sha256_computed':False}
    output = work / 'notebook_verification.json'
    output.write_text(json.dumps(evidence, ensure_ascii = False, indent = 2), encoding = 'utf-8')
    print(json.dumps(evidence, ensure_ascii = False, indent = 2))


if __name__ == '__main__':
    main()
