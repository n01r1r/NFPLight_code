import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'id': f'cell-{len(cells):03d}', 'metadata': {},
                'source': source.strip().splitlines(keepends = True)}
        if kind == 'code':
            cell.update(execution_count = None, outputs = [])
        cells.append(cell)
    def section(title, source):
        add('markdown', '## ' + title)
        add('code', source)

    section('00 · Clone / Revision', '''
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys

REPOSITORY = 'https://github.com/n01r1r/NFPLight_code.git'
REVISION = 'codex/colab-experiments'
BASE = Path('/content') if Path('/content').is_dir() else Path.cwd()
PROJECT = BASE / 'NFPLight_code'
if not PROJECT.exists():
    subprocess.check_call(['git', 'clone', '--depth', '1', '--branch', REVISION, REPOSITORY, str(PROJECT)])
elif not (PROJECT / '.git').exists():
    raise FileExistsError(f'Choose an empty project directory: {PROJECT}')
remote = subprocess.check_output(['git', '-C', str(PROJECT), 'remote', 'get-url', 'origin'], text=True).strip()
if remote.removesuffix('.git').lower() != REPOSITORY.removesuffix('.git').lower():
    raise RuntimeError(f'Existing clone has another remote: {remote}')
commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
expected = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', REVISION], text=True).strip()
if commit != expected or subprocess.run(['git', '-C', str(PROJECT), 'diff', '--quiet', 'HEAD']).returncode:
    raise RuntimeError('Existing clone differs from requested revision; use a separate clean clone')
for name, module in list(sys.modules.items()):
    if name in ('colab_tools', 'serve', 'live_training_plot', 'train_colab') or name == 'nfplight' or name.startswith('nfplight.'):
        location = getattr(module, '__file__', None)
        if location is not None and not Path(location).resolve().is_relative_to(PROJECT.resolve()):
            raise RuntimeError(f'{name} is loaded from another clone; restart the runtime')
        del sys.modules[name]
importlib.invalidate_caches()
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print({'project': str(PROJECT), 'commit': commit, 'revision': REVISION})
''')
    section('01 · Dependencies', '''
import importlib.metadata
loaded_versions = {name: getattr(sys.modules[name], '__version__', None)
                   for name in ('einops', 'numpy', 'psutil', 'matplotlib', 'tqdm', 'packaging') if name in sys.modules}
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'])
for name, version in loaded_versions.items():
    if version is not None and version != importlib.metadata.version(name):
        raise RuntimeError(f'{name} changed after import; restart the runtime and rerun from the top')
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import torch
if Version(torch.__version__) not in SpecifierSet('>=2.1,<3'):
    raise RuntimeError(f'CUDA PyTorch >=2.1,<3 required: {torch.__version__}')
get_ipython().run_line_magic('matplotlib', 'inline')
print({'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
       'packages': {name: importlib.metadata.version(name) for name in ('einops', 'numpy', 'psutil', 'matplotlib', 'tqdm')}})
''')
    section('02 · Model / Relative paths / Controls', '''
PROFILES = {'nfplight': 'configs/nfplight_32_noquant.json', 'naf_unetpp': 'configs/nested_32_noquant.json'}
MODEL = 'naf_unetpp'
MODEL_CONFIG = Path(PROFILES[MODEL])
TRAINING_CONFIG = Path('configs/training_noquant.json')
RUNTIME_CONFIG = Path('configs/runtime_accum4.json')
DRIVE_MOUNT = Path('/content/drive')
DRIVE_DATA_FOLDER = Path('MyDrive/MatSynth_PRIMA_Fabric')
MOUNT_DRIVE = True
DATA_ROOT = Path('data/MatSynth_PRIMA_Fabric')
SOURCE_MANIFEST = DATA_ROOT / 'manifests/fabric_leather_training_manifest.jsonl'
MANIFEST = Path('work/train_val_manifest.jsonl')
RUN_NAME = MODEL + '_fresh_01'
RUN_ROOT = Path('runs') / RUN_NAME
RESUME = None
RUN_PIPELINE_PREVIEW = False
RUN_TRAINING = False
SHOW_PLOTS = True
REFRESH_SECONDS = 10
Path('work').mkdir(exist_ok=True)
''')
    section('03 · Drive / Relative dataset link', '''
from colab_tools import relative_path, attach_drive_dataset
relative_path(DRIVE_DATA_FOLDER)
relative_path(DATA_ROOT)
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT))
drive_source = (DRIVE_MOUNT / DRIVE_DATA_FOLDER).resolve()
if not drive_source.is_relative_to(DRIVE_MOUNT.resolve()):
    raise ValueError('Drive dataset folder escapes mount')
attach_drive_dataset(drive_source, DATA_ROOT)
print({'drive_source':str(drive_source), 'relative_dataset':str(DATA_ROOT)})
''')
    section('04 · Dataset / Experiment contract', '''
from colab_tools import relative_path, relocate_manifest
from nfplight.checkpoint import read_config
for path in (MODEL_CONFIG, TRAINING_CONFIG, RUNTIME_CONFIG, DATA_ROOT, SOURCE_MANIFEST, MANIFEST, RUN_ROOT):
    relative_path(path)
if RESUME is not None:
    relative_path(RESUME)
inventory = relocate_manifest(SOURCE_MANIFEST, DATA_ROOT, MANIFEST, expected_counts={
    'train/Fabric':358, 'train/Leather':208, 'val/Fabric':98, 'val/Leather':66})
profile = read_config(MODEL_CONFIG)
runtime = json.loads(RUNTIME_CONFIG.read_text())
policy = json.loads(TRAINING_CONFIG.read_text())
print(json.dumps({'dataset':inventory, 'profile':profile, 'runtime':runtime, 'training':policy}, indent=2))
''')
    section('05 · NAFBlock / Selected architecture', '''
import inspect
from IPython.display import Code, display
from nfplight.network import NAFBlock, NFPLight
from nfplight.nested_model import NestedDecoder, NestedModel
display(Code(inspect.getsource(NAFBlock), language='python'))
if profile['network'].get('architecture', 'nfplight') == 'nested':
    display(Code(inspect.getsource(NestedDecoder), language='python'))
    display(Code(inspect.getsource(NestedModel), language='python'))
else:
    display(Code(inspect.getsource(NFPLight), language='python'))
''')
    section('06 · Input / Loss / Training pipeline', '''
from nfplight.features import Features
from nfplight.pipeline import TrainingPipeline
display(Code(inspect.getsource(Features), language='python'))
display(Code(inspect.getsource(TrainingPipeline), language='python'))
''')
    section('07 · Model / Pipeline plot', '''
from colab_tools import plot_pipeline
import matplotlib.pyplot as plt
if SHOW_PLOTS:
    figure = plot_pipeline(profile)
    plt.show()
''')
    section('08 · CUDA pipeline / Module shapes', '''
if RUN_PIPELINE_PREVIEW:
    preview_out = Path('work') / (RUN_NAME + '_pipeline')
    subprocess.run([sys.executable, 'preview_pipeline.py', '--manifest', str(MANIFEST), '--out', str(preview_out),
                    '--config', str(MODEL_CONFIG), '--training-config', str(TRAINING_CONFIG), '--runtime', str(RUNTIME_CONFIG)], check=True)
    print((preview_out / 'shapes.json').read_text())
    if SHOW_PLOTS:
        from IPython.display import Image
        display(Image(filename=str(preview_out / 'features.png')))
''')
    section('09 · CUDA FP32 training / tqdm / Live plots', '''
from colab_tools import monitor
if RUN_TRAINING:
    if RUN_ROOT.exists():
        raise FileExistsError(RUN_ROOT)
    active_child = globals().get('training_process')
    if active_child is not None and active_child.poll() is None:
        raise RuntimeError('This notebook already owns a running trainer; use a separate runtime for another model')
    command = [sys.executable, '-u', 'train_colab.py', '--run-root', str(RUN_ROOT), '--manifest', str(MANIFEST),
               '--config', str(MODEL_CONFIG), '--training-config', str(TRAINING_CONFIG), '--runtime', str(RUNTIME_CONFIG)]
    if RESUME is not None:
        command.extend(['--resume', str(RESUME)])
    stdout_path = Path('work') / (RUN_NAME + '.stdout.log')
    stderr_path = Path('work') / (RUN_NAME + '.stderr.log')
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        training_process = subprocess.Popen(command, cwd=PROJECT, stdout=stdout, stderr=stderr)
    import time
    while training_process.poll() is None and not (RUN_ROOT / 'process.json').exists():
        time.sleep(.2)
    if training_process.poll() is not None and not (RUN_ROOT / 'process.json').exists():
        raise RuntimeError(stderr_path.read_text()[-12000:])
    try:
        final_snapshot = monitor(RUN_ROOT, training_process, refresh_seconds=REFRESH_SECONDS, plots=SHOW_PLOTS)
    except BaseException:
        print({'child_pid':training_process.pid, 'still_running':training_process.poll() is None,
               'stderr':str(stderr_path), 'last_checkpoint_exists':(RUN_ROOT / 'run/last.pt').is_file()})
        raise
    return_code = training_process.wait()
    if return_code:
        raise RuntimeError(stderr_path.read_text()[-12000:])
    print(json.dumps(final_snapshot['status'], indent=2))
''')
    section('10 · Reopen plots / Snapshot', '''
if (RUN_ROOT / 'run/train.jsonl').exists():
    active_child = globals().get('training_process')
    snapshot = monitor(RUN_ROOT, active_child if active_child is not None and active_child.poll() is None else None,
                       refresh_seconds=REFRESH_SECONDS, plots=SHOW_PLOTS)
    print(json.dumps({'status':snapshot['status'], 'latest':snapshot['latest'], 'validation':snapshot['latest_validation']}, indent=2))
    from IPython.display import FileLink
    display(FileLink(str(RUN_ROOT / 'reports/snapshot.json')))
    if (RUN_ROOT / 'reports/training.png').is_file():
        display(FileLink(str(RUN_ROOT / 'reports/training.png')))
''')
    section('11 · Stop child / Preserve checkpoint', '''
STOP_TRAINING = False
if STOP_TRAINING and 'training_process' in globals() and training_process.poll() is None:
    import signal
    training_process.send_signal(signal.SIGINT)
    exit_code = training_process.wait()
    last_checkpoint = RUN_ROOT / 'run/last.pt'
    print({'exit_code':exit_code, 'last_checkpoint':str(last_checkpoint) if last_checkpoint.is_file() else None})
''')
    notebook = {'cells':cells, 'metadata': {'kernelspec':{'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info':{'name':'python'}, 'colab':{'name':'train.ipynb'}, 'accelerator':'GPU',
        'nfplight':{'code_source':'git clone', 'data_included':False, 'weights_included':False,
                    'design_reference':'https://github.com/hun9008/HoViT', 'automatic_training':False}},
        'nbformat':4, 'nbformat_minor':5}
    path = ROOT / 'notebooks/train.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii = False, indent = 1) + '\n', encoding = 'utf-8')
    print({'notebook':str(path), 'cells':len(cells), 'bytes':path.stat().st_size})


if __name__ == '__main__':
    build()
