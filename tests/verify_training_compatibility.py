import argparse
import sys
from pathlib import Path
import json
import gc
import copy
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--legacy-root', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--checkpoint')
args = parser.parse_args()
if Path(args.out).exists():
    raise FileExistsError('verification output already exists')
OLD = Path(args.legacy_root).resolve()
sys.path[:0] = [str(ROOT),str(OLD)]
from verify_compatibility import equal
from nfplight.checkpoint import read_config,load_payload,state_dict
from nfplight.network import NFPLight
from nfplight.features import Features
from nfplight.rendering import Renderer
from nfplight.data import MaterialBank,canonical_target,fixed_f0
from nfplight.storage import write_json,identity
from network.nfplight_net import TwoBranchNet,NAFBlock
from model.nfplight_model import NFPLightModel
from benchmark_fabric_leather import Bank,prepare

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(19)
profile=read_config(ROOT/'configs/7m_c.json')
checkpoint=Path(args.checkpoint) if args.checkpoint else next((OLD/'artifacts/weight_handoff_20261019/weights').glob('03*.pt'))
checkpoint_identity=identity(checkpoint)
payload=load_payload(checkpoint)
if identity(checkpoint)!=checkpoint_identity:
    raise ValueError('checkpoint changed during read')
weights=state_dict(payload)
old=TwoBranchNet(width=32,middle_blk_num=4,enc_blk_nums=[2,2,4],dec_blk_nums=[2,2,2])
for module in old.modules():
    if isinstance(module,NAFBlock):
        module.dropout1=torch.nn.Dropout(.1)
        module.dropout2=torch.nn.Dropout(.1)
new=NFPLight(**profile['network'])
old.load_state_dict(weights);new.load_state_dict(weights)
del weights
old.to(args.device);new.to(args.device)
optimizers=[torch.optim.Adam(net.parameters(),lr=1e-5) for net in (old,new)]
if args.checkpoint:
    for optimizer in optimizers:
        optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
x=torch.rand(1,21,32,32,device=args.device)*2-1
rng=torch.random.get_rng_state()
cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else []
if args.checkpoint:
    rng=payload['rng']['torch']
    cuda_rng=payload['rng']['cuda'] if args.device == 'cuda' else []
del payload
for net,opt in zip((old,new),optimizers):
    torch.random.set_rng_state(rng)
    if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
    result=net(x)
    loss=(result-.2).abs().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    opt.step()
equal(old.state_dict(),new.state_dict(),'actual7m Adam update')
equal(optimizers[0].state_dict(),optimizers[1].state_dict(),'actual7m Adam moments')
for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
    equal(a.grad,b.grad,'gradient '+name)
print('PASS actual 7M checkpoint: dropout, gradients, Adam weights/moments exact',flush=True)
del old,new,optimizers,result,loss
gc.collect()
execution=json.loads((OLD/'artifacts/C_convergence_followup_20261019/run/execution_config.json').read_text())
original=Bank(execution['manifest'],'train')
bank=MaterialBank(execution['manifest'],'train',cache_size=4,f0=.04,crop_count=16)
for rotation in range(4):
    for flip in (False,True):
        equal(original.crop(0,0,rotation,flip),bank.crop(0,0,rotation,flip),'D4 bank')
class Reference(NFPLightModel):
    def init_network(self):pass
reference=Reference(SimpleNamespace(device=args.device,feature_version='raw_calibrated_gain9_v3',capture_precision=True))
target=bank.crop(0,0)[None].to(args.device)
safe,mask,expected=prepare(reference,target)
runtime=json.loads((ROOT/'configs/runtime.json').read_text())
renderer=Renderer(runtime['renderer'],device=args.device)
features=Features(profile['features'],device=args.device)
actual_safe,actual_mask=canonical_target(target,roughness_floor=.05)
actual_safe=fixed_f0(actual_safe,.04)
pair=renderer.observations(actual_safe,4.,12.)*actual_mask
actual=features(pair,purpose='synthetic',mask=actual_mask)[0]
equal(safe,actual_safe,'target');equal(mask,actual_mask,'valid mask');equal(expected,actual,'real dataset features')
print('PASS actual dataset crop: all D4 transforms, safe target, mask and training features exact',flush=True)
write_json(Path(args.out),{'passed':True,'device':args.device,'dataset_device':args.device,'actual_7m_backward_and_adam_exact':True,'checkpoint':checkpoint_identity,
    'restored_optimizer_and_rng':bool(args.checkpoint),'network_input_shape':[1,21,32,32],
    'actual_dataset_crop_d4_and_features_exact':True,'material_id':bank.records[0]['material_id'],
    'research_training_started':False,'gpu_production_next_step_tested':False,'sha256_computed':False})
