from experiments import map_factor_followup as f
from experiments.robustness_suite import config
from experiments.robustness_suite.run import create_network
import torch

def test_fresh_policy_and_initialization(monkeypatch):
 monkeypatch.setitem(config.ARMS,f.ARM,{'parameters':f.PARAMETERS,'middle_blocks':4,'extra_training_crops':True,'factor':.8,'render_cap':25000})
 old=config.policy_for_arm('extra_crop7m_p01');new=config.policy_for_arm(f.ARM)
 assert {k for k in old if old[k]!=new[k]}=={'arm','factor','render_cap'}
 backbone={'width':32,'middle_blocks':4,'encoder_blocks':[2,2,4],'decoder_blocks':[2,2,2]}
 torch.manual_seed(19);a=create_network(old,backbone)
 torch.manual_seed(19);b=create_network(new,backbone)
 assert sum(p.numel() for p in b.parameters())==f.PARAMETERS
 assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
 for initial,floor in [(5e-4,1e-5),(1e-4,1e-6)]:
  optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))],lr=initial)
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=new['factor'],patience=5,cooldown=2,min_lr=floor,threshold=.005)
  scheduler.step(1.)
  for _ in range(6):scheduler.step(1.1)
  assert optimizer.param_groups[0]['lr']==initial*.8
