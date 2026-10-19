"""Render-only semantics, matched streams, predecessor gate and joint decision."""
import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch
from experiments.render_only_followup import run, comparison, real_gallery
from experiments.full_dataset_suite.engine import Engine, capture_rng, restore_rng
from experiments.full_dataset_suite.config import policy_for_arm


def test_map_gradient_is_absent_and_render_gradient_preserved():
    m=torch.tensor(2.,requires_grad=True)
    r=torch.tensor(3.,requires_grad=True)
    loss=run.render_objective(r.square(),m.square())
    a,b=torch.autograd.grad(loss,(m,r),allow_unused=True)
    assert a is None and b.item()==6. and loss.item()==9.


def test_actual_pipeline_detaches_diagnostics(monkeypatch):
    pred=torch.ones(1,10,1,1,requires_grad=True)
    seen=[]
    monkeypatch.setattr(run,'prepare',lambda m,t:(t,torch.ones_like(t,dtype=torch.bool),t))
    monkeypatch.setattr(run,'unit_prediction',lambda x:x)
    def maps(p,t,v):
        seen.append(p.requires_grad)
        parts=tuple((p+i).mean() for i in range(4))
        return sum(parts),parts
    monkeypatch.setattr(run,'ndrs_loss',maps)
    monkeypatch.setattr(run,'render_loss',lambda m,p,t,v,l:p.square().mean())
    pipe=object.__new__(run.RenderOnlyPipeline)
    pipe.model=None;pipe.training_lights=None
    pipe.train=SimpleNamespace(crop=lambda *args:torch.zeros(10,1,1))
    loss,metrics=pipe.objective(lambda f:pred,[0],[0],'render')
    loss.backward()
    assert seen==[False]
    torch.testing.assert_close(pred.grad,torch.full_like(pred,.2))
    assert metrics['map_weight']==0. and metrics['map_loss']==10.
    with pytest.raises(ValueError,match='cannot perform map'):
        pipe.objective(lambda f:pred,[0],[0],'map')


def test_fresh_initialization_matches_batch_d4_dropout_cpu(tmp_path):
    original=capture_rng()
    try:
        torch.manual_seed(19)
        net=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
        best={'net_g':copy.deepcopy(net.state_dict()),'stage':'map','step':135000,'ndrs_sum':.12,'render_l1':.0034}
        end={'rng':capture_rng(),'sample_rng':random.Random(1).getstate(),'d4_rng':random.Random(2).getstate(),
             'augmentation_rng':random.Random(3).getstate(),'noise_rng':torch.Generator().manual_seed(4).get_state()}
        observed=[]
        for arm in ('B','D'):
            policy=policy_for_arm('baseline7m_p01');policy.update(arm=arm,render_lr=1e-5)
            engine=Engine(tmp_path/arm,copy.deepcopy(net),policy,{},None,None,sample_count=9056)
            run.initialize_render(engine,best,end)
            assert not engine.optimizer.state and engine.optimizer.param_groups[0]['lr']==1e-5
            observed.append((engine.samples.sample(range(9056),4),[engine.d4.randrange(8) for _ in range(4)],
                             engine.net(torch.ones(4,2)).detach()))
        assert observed[0][:2]==observed[1][:2]
        torch.testing.assert_close(observed[0][2],observed[1][2],rtol=0,atol=0)
    finally:
        restore_rng(original)


def test_reference_log_exact_length_and_inputs(tmp_path):
    row={'step':1,'indices':[1,2,3,4],'d4_transforms':[0,1,2,3]}
    a,b=tmp_path/'a',tmp_path/'b'
    a.write_text(json.dumps(row)+'\n');b.write_text(a.read_text())
    assert run.compare_logs(a,b,count=1)==1
    b.write_text(a.read_text()*2)
    with pytest.raises(ValueError,match='length'):run.compare_logs(a,b,count=1)
    row['d4_transforms'][0]=7;b.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='sequence'):run.compare_logs(a,b,count=1)


def test_wait_for_training_and_real_completion_and_stop_on_failure():
    assert not run.prerequisites({'status':'running'},{'status':'complete'})
    assert not run.prerequisites({'status':'complete'},{'status':'running'})
    assert run.prerequisites({'status':'complete'},{'status':'complete'})
    with pytest.raises(RuntimeError):run.prerequisites({'status':'stopped'},{'status':'complete'})
    with pytest.raises(RuntimeError):run.prerequisites({'status':'complete'},{'status':'failed'})


def test_joint_decision_does_not_hide_map_component_tradeoff():
    baseline={k:1. for k in comparison.KEYS}
    selected={k:dict(baseline) for k in ('B','C','D')}
    selected['B']['render_l1']=.95;selected['C']['render_l1']=.9
    selected['D']={k:.8 for k in comparison.KEYS}
    final=copy.deepcopy(selected)
    assert comparison.decide(baseline,selected,final)['decision']=='promote_baseline_vs_render_only_screening'
    selected['D']['map_specular']=1.01
    assert comparison.decide(baseline,selected,final)['decision']=='render_gain_with_tradeoff_keep_primary_pending'
    selected['D']['render_l1']=.99
    assert comparison.decide(baseline,selected,final)['decision']=='keep_existing_primary_render_only_not_superior_in_screen'


def test_gallery_replacement_preserves_original_and_is_idempotent():
    text='<h1>old</h1><p>original results</p>'
    section=real_gallery.START+'new'+real_gallery.END
    result=real_gallery.replace_section(text,section)
    assert '<p>original results</p>' in result
    assert real_gallery.replace_section(result,section)==result
    newer=real_gallery.replace_section(result,real_gallery.START+'complete'+real_gallery.END)
    assert 'new' not in newer and 'complete' in newer


def test_group_pairing_uses_groups_not_correlated_folders():
    a=[{'group_id':g,**{k:v for k in comparison.KEYS}} for g,v in [('a',1.),('a',3.),('b',4.)]]
    b=[{'group_id':g,**{k:2. for k in comparison.KEYS}} for g in ['a','b']]
    result=comparison.paired_groups(a,b)['render_l1']
    assert result['groups']==2 and result['D_minus_reference_mean']==1.
    assert result['D_higher_groups']==1


def test_prepared_contract_cannot_change_after_worker_started(tmp_path,monkeypatch):
    suite=tmp_path/'abc';out=tmp_path/'d';out.mkdir()
    ref=suite/'runs'/run.REFERENCES[0];ref.mkdir(parents=True)
    config={'policy':{},'scientific_settings':{'loss':{}},'followup':{},'file_identities':[]}
    (ref/'execution_config.json').write_text(json.dumps(config))
    (ref/'display_history.json').write_text('{}')
    monkeypatch.setattr(run,'SUITE',suite);monkeypatch.setattr(run,'OUT',out);monkeypatch.setattr(run,'RUN',out/'run')
    monkeypatch.setattr(run,'validate_sources',lambda c:None)
    run.prepare_run()
    path=out/'run/execution_config.json';saved=json.loads(path.read_text())
    saved['policy']['render_lr']=123;path.write_text(json.dumps(saved))
    (out/'queue.json').write_text('{}')
    with pytest.raises(ValueError,match='Started render-only contract changed'):
        run.prepare_run()


def test_comparison_builds_joint_artifacts_and_rejects_changed_population(tmp_path,monkeypatch):
    suite=tmp_path/'abc';out=tmp_path/'d';out.mkdir()
    monkeypatch.setattr(comparison,'SUITE',suite);monkeypatch.setattr(comparison,'OUT',out)
    names={'A':'A_weight1_lr1e4','B':'B_weight1_lr1e5','C':'C_weight10_lr1e5'}
    for label,factor in [('A',.95),('B',.9),('C',.8),('D',.7)]:
        folder=out/'run' if label=='D' else suite/'runs'/names[label]
        folder.mkdir(parents=True)
        entries=[]
        for step in range(0,10001,1000):
            v=1. if step==0 else factor
            mean={k:v for k in comparison.KEYS};mean['ndrs_sum']=4*v
            materials=[{'group_id':f'g{i//2}','material_id':str(i),'category':'Fabric' if i<98 else 'Leather',**mean} for i in range(164)]
            entries.append({'stage':'render','step':step,'populations':{'heldout':{
                'mean':mean,'materials':materials,'sample_folders':164,'crop_indices':[0,4,8,12],'novel_lights':12}}})
        (folder/'evaluation.json').write_text(json.dumps(entries))
        (folder/'complete.json').write_text(json.dumps({'status':'screening_complete','render_updates':10000,
            'selected':'best_render','selection_step':10000,'render_l1':factor}))
    (out/'run/matching_verification.json').write_text('{"passed":true}')
    result=comparison.build()
    assert result['decision']['decision']=='promote_baseline_vs_render_only_screening'
    assert (out/'COMPARISON_KO.md').is_file() and (out/'validation_trajectories.csv').is_file()
    assert len((out/'validation_trajectories.csv').read_text().splitlines())==45
    entries[-1]['populations']['heldout']['crop_indices']=[0]
    (out/'run/evaluation.json').write_text(json.dumps(entries))
    with pytest.raises(ValueError,match='heldout contract'):
        comparison.build()
