import json
import torch
from train_fabric_leather import atomic_checkpoint, truncate_log_to_checkpoint
from live_training_plot import LogReader


def test_atomic_checkpoint_and_log_rollback(tmp_path):
    path=tmp_path/'last.pt'
    atomic_checkpoint(path,{'step':2,'tensor':torch.tensor([1.,2.])})
    value=torch.load(path,weights_only=True)
    assert value['step']==2
    assert not (tmp_path/'last.pt.tmp').exists()
    log=tmp_path/'train.jsonl'
    log.write_text(''.join(json.dumps({'total_step':i})+'\n' for i in (1,2,3))+'{"total_step":',encoding='utf-8')
    truncate_log_to_checkpoint(log,2)
    assert [json.loads(line)['total_step'] for line in log.read_text().splitlines()]==[1,2]


def test_viewer_uses_recorded_global_steps_and_categories(tmp_path):
    row={'stage':'render','step':1000,'total_step':401000,'render_l1':.1,
         'category':{'Leather':{'normal_degrees':12.,'render_l1':.2}}}
    (tmp_path/'validation.json').write_text(json.dumps({'history':[row]}),encoding='utf-8')
    (tmp_path/'status.json').write_text(json.dumps({'status':'training'}),encoding='utf-8')
    result=LogReader(tmp_path).snapshot()
    assert result['val']['render:render_l1']==[[401000,.1]]
    assert result['val']['render:Leather:normal_degrees']==[[401000,12.]]
    assert result['status']['status']=='training'
