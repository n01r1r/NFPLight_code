"""Actual weighted gradients and matched-start render state, without CUDA."""
import copy
import random
import pytest
import torch
from experiments.full_dataset_suite.config import policy_for_arm
from experiments.full_dataset_suite.engine import Engine, capture_rng
from experiments.render_weight_followup.run import combine_loss, initialize_render


def test_weight_changes_render_gradient_and_leaves_map_gradient():
    m=torch.tensor(2.,requires_grad=True)
    r=torch.tensor(3.,requires_grad=True)
    loss=combine_loss(m.square(),r.square(),10.)
    loss.backward()
    assert loss.item()==94
    assert m.grad.item()==4 and r.grad.item()==60


@pytest.mark.parametrize('weight',[0.,-1.,float('inf'),float('nan')])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError): combine_loss(torch.tensor(1.),torch.tensor(2.),weight)


def test_same_checkpoint_rng_fresh_adam_and_render_only(tmp_path):
    saved=capture_rng()
    end={'rng':saved,'sample_rng':random.Random(42).getstate(),'d4_rng':random.Random(43).getstate(),
         'augmentation_rng':random.Random(44).getstate(),'noise_rng':torch.Generator().manual_seed(45).get_state()}
    base=torch.nn.Linear(2,1)
    best={'net_g':copy.deepcopy(base.state_dict()),'stage':'map','step':135000,'total_step':135000,
          'ndrs_sum':.12,'render_l1':.0034,'metric':.12}
    states=[]
    for i,lr in enumerate([1e-4,1e-5]):
        policy=policy_for_arm('baseline7m_p01');policy['render_lr']=lr
        model=torch.nn.Linear(2,1)
        engine=Engine(tmp_path/str(i),model,policy,{},None,None,sample_count=9056)
        initialize_render(engine,best,end)
        assert engine.state['stage']=='render' and engine.state['stage_step']==0
        assert engine.state['total_step']==135000 and engine.state['new_map_updates']==0
        assert not engine.optimizer.state and engine.optimizer.param_groups[0]['lr']==lr
        assert all(torch.equal(v,best['net_g'][k]) for k,v in model.state_dict().items())
        states.append((engine.samples.sample(range(9056),4),[engine.d4.randrange(8) for _ in range(4)],torch.rand(8)))
    assert states[0][:2]==states[1][:2] and torch.equal(states[0][2],states[1][2])


def test_wrong_start_weight_rejected(tmp_path):
    with pytest.raises(ValueError,match='map135000'):
        initialize_render(None,{'stage':'map','step':134000},{})


def test_report_reuses_template_and_embeds_all_fixed_panels(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.render_weight_followup import report
    from experiments.full_dataset_suite.reporting import write_offline_report
    from experiments.full_dataset_suite.config import ROOT
    run=tmp_path/'condition';run.mkdir()
    out=run/'report';out.mkdir()
    # No GPU: test the HTML transformation independently of expensive evaluation.
    def offline(path):
        text=(ROOT/'live_training_plot.html').read_text(encoding='utf-8')
        text=text.replace("await fetch('/api',{cache:'no-store'})",'({ok:true,json:async()=>({})})').replace('setInterval(refresh,10000);','')
        (path/'offline_report.html').write_text(text,encoding='utf-8')
    monkeypatch.setattr(report,'write_offline_report',offline)
    uri='data:image/png;base64,iVBORw0KGgo='
    panels={str(i):{k:[uri]*5 for k in ['GT','best-map','best-render']} for i in range(12)}
    base=SimpleNamespace(OUT=out,table=lambda rows:'<table></table>')
    report.export(base,run,{}, {},panels,{})
    text=(out/'index.html').read_text(encoding='utf-8')
    assert text.count('src="data:image/png;base64,')==180
    assert 'train-table' in text and 'test-qualitative' in text
    assert 'fetch(' not in text and 'setInterval(' not in text
