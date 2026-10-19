"""Cross-entrypoint checks of the shared spatial relation; no dataset hashes."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from model.nfplight_model import NFPLightModel, CURRENT_RAW_FEATURE_VERSION
from model.nfplight_real_model import NFPLightModel as RealModel
from model.nfplight_matsynth_capture_model import MatSynthCaptureModel
from model.author_real_capture_adapter import assemble_legacy33_features, legacy_geometry
from model.relation_geometry import RELATION_VERSION
from utils import svBRDF


def model(device='cpu'):
    m = NFPLightModel.__new__(NFPLightModel)
    m.feature_version = CURRENT_RAW_FEATURE_VERSION
    m.device = torch.device(device)
    m.capture_precision = False  # Match real33 angular setup for exact cross-path comparison.
    m.renderer = svBRDF()
    m.init_distance()
    m.init_rendering()
    return m


def pair(device='cpu'):
    x = torch.rand((1,6,256,256), generator=torch.Generator().manual_seed(42)) * .02
    x[:,:3] *= 8
    return x.to(device)


def test_train_eval_and_capture_relation_match_real33_with_same_gain():
    m = model(); x = pair()
    for capture in (False, True):
        f = (m.build_capture_features(x) if capture else m.build_features(x))[0]
        f33, trace = assemble_legacy33_features(x,x,*legacy_geometry(256,'cpu'),
                                               fixed_gain=None if capture else 9.)
        torch.testing.assert_close(f[:,18:20], f33[:,30:32], rtol=0, atol=0)
        torch.testing.assert_close(trace['spatial_attenuation_factor'], m.spatial_attenuation_factor, rtol=0, atol=0)
    assert not torch.equal(m.build_features(x)[0][:,18:20], m.build_capture_features(x)[0][:,18:20])
    c = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
    c.__dict__.update(m.__dict__); c.args=SimpleNamespace(); c.net_g=Recorder(); c.normal_head='xyz'
    c.feed_data({'inputs':x,'name':['fixture']}); c.test()
    torch.testing.assert_close(c.net_g.features, m.build_capture_features(x)[0], rtol=0, atol=0)
    r = RealModel.__new__(RealModel); r.args=SimpleNamespace(image_size=256); r.device=torch.device('cpu')
    r.init_distance();r.init_rendering();r.net_g=Recorder()
    r.feed_data({'inputs':x,'name':['fixture']});r.test()
    expected = assemble_legacy33_features(x,x,*legacy_geometry(256,'cpu'))[0]
    torch.testing.assert_close(r.net_g.features, expected, rtol=0, atol=0)


class Recorder(nn.Module):
    def forward(self, features):
        self.features=features
        return torch.zeros((features.shape[0],10,256,256),device=features.device)


def test_batch_mask_and_independent_clipping_formula():
    m=model();x=torch.cat([pair(),pair()*.7]);x[:,3,0,0]=1
    valid=torch.ones((2,1,256,256),dtype=torch.bool);valid[:,:,:8,:8]=False
    f=m.build_features(x,valid)[0]
    for i in range(2):
        torch.testing.assert_close(f[i:i+1],m.build_features(x[i:i+1],valid[i:i+1])[0],rtol=0,atol=0)
    changed=x.clone();changed[:,:,:8,:8]=1
    assert torch.equal(f,m.build_features(changed,valid)[0])
    assert (f[:,:,:8,:8]==-1).all()
    axis=(torch.arange(256,dtype=torch.float32)-127.5)/128
    rho=axis[:,None]**2+axis[None,:]**2
    ratio=(rho+144)/(rho+16);factor=(ratio/ratio[127,127])[None,None]
    clean=x*valid
    q=((m.time_co_map*(clean[:,3:]*9*factor)).clamp(0,1)-clean[:,:3]).abs().mean(1,keepdim=True)/m.coefficient.clamp_min(1e-5)
    expected=(1-m.ClipToOne(q,valid))*valid
    torch.testing.assert_close(f[:,18:19],expected*2-1,rtol=0,atol=0)
    assert torch.equal(m.spatial_attenuation_factor,factor)


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_actual_train_forward_backward_and_eval(device):
    from train_fabric import _prepare_target
    from network.nfplight_net import TwoBranchNet
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    m=model(device)
    m.net_g=TwoBranchNet(width=4,middle_blk_num=1,enc_blk_nums=[1,1],dec_blk_nums=[1,1]).to(device)
    target=torch.zeros(1,10,256,256,device=device);target[:,2]=1;target[:,7:]=-.92
    safe,valid,rendered,features=_prepare_target(m,target)
    assert features.dtype==torch.float32 and torch.isfinite(features).all()
    assert torch.equal(features,m.build_features(rendered,valid)[0])
    m.net_g.train();opt=torch.optim.Adam(m.net_g.parameters(),lr=1e-4)
    pred=m.net_g(features);loss=(pred-safe).square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.net_g.parameters() if p.grad is not None)
    opt.step();m.net_g.eval()
    with torch.no_grad():
        assert torch.isfinite(m.net_g(features)).all()


def test_report_new_trace_exposes_spatial_formula_without_relabeling_old():
    from capture_processing.report import _make_html
    manifest={'group':'test','conditions':{'both':{'arrays':{}}},'report':{},
              'model_contract':{'estimator_family':'author_legacy33','feature_channels':33}}
    old=_make_html(manifest,[],{},{},{},[])
    manifest['conditions']['both']['arrays']['spatial_attenuation_factor']={}
    new=_make_html(manifest,[],{},{},{},[])
    assert 'K_L K_θ F' in new and 'K_L K_θ F' not in old
    assert 'r_F²/r_N²' in new


def test_training_checkpoint_records_version_and_blocks_old_resume(tmp_path):
    from train_fabric import save_checkpoint,load_checkpoint
    import random
    m=model();m.net_g=nn.Conv2d(21,10,1);opt=torch.optim.Adam(m.net_g.parameters())
    path=tmp_path/'checkpoint.pth'
    save_checkpoint(path,m,{}, {},optimizer=opt,rng=random.Random(1))
    payload=torch.load(path,weights_only=False)
    assert payload['relation_version']==RELATION_VERSION
    load_checkpoint(path,m,{},optimizer=opt,rng=random.Random(1),payload=payload)
    payload.pop('relation_version')
    with pytest.raises(ValueError,match='relation_version'):
        load_checkpoint(path,m,{},optimizer=opt,rng=random.Random(1),payload=payload)


def test_real_evaluator_runs_shared_features_and_preserves_mode_and_rng():
    from train_fabric import evaluate
    m=model();m.net_g=nn.Sequential(nn.Conv2d(21,10,1),nn.Tanh())
    target=torch.zeros(10,256,256);target[2]=1;target[7:]=-.92
    class Dataset:
        manifest={'crop_count':2}
        records=[{'id':'synthetic'}]
        def get_crop(self,index,crop):
            return target.clone()
    before=torch.get_rng_state().clone()
    first=evaluate(m,Dataset(),crop_count=2,light_count=1,eval_batch_size=2)
    second=evaluate(m,Dataset(),crop_count=2,light_count=1,eval_batch_size=1)
    assert m.net_g.training and torch.equal(before,torch.get_rng_state())
    for key,value in first['mean'].items():
        assert value==pytest.approx(second['mean'][key],abs=1e-5)


def test_grayscale_runner_rejects_pre_spatial_baseline_before_creating_output(tmp_path):
    import json
    from experiments.run_relation_grayscale_comparison import run
    (tmp_path/'run_real_comparison.py').write_text("CAPTURE_GROUPS=['fixture']",encoding='utf-8')
    (tmp_path/'fixture').mkdir()
    (tmp_path/'fixture/manifest.json').write_text(json.dumps({'model_contract':{}}),encoding='utf-8')
    with pytest.raises(ValueError,match='relation_version'):
        run(tmp_path)
    assert not (tmp_path/'relation_comparison').exists()
