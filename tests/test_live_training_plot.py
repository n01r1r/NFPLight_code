import json
from pathlib import Path
from live_training_plot import LogReader, plateau


def test_inherited_map_history_stops_at_selected_weight_and_keeps_current_status(tmp_path):
    parent = tmp_path/'parent'
    arm = tmp_path/'arm'
    parent.mkdir()
    arm.mkdir()
    records = [{'stage': 'map', 'step': i, 'total_step': i, 'loss': float(i)} for i in (1, 2, 3)]
    records.append({'stage': 'render', 'step': 1, 'total_step': 4, 'loss': 99.})
    original = ''.join(json.dumps(row)+'\n' for row in records)
    (parent/'train.jsonl').write_text(original)
    (parent/'validation.json').write_text(json.dumps({'history': [
        {'stage': row['stage'], 'step': row['step'], 'total_step': row['total_step'], 'render_l1': row['loss']}
        for row in records]}))
    (arm/'display_history.json').write_text(json.dumps({'source_run': '../parent', 'stage': 'map', 'through_total_step': 2}))
    (arm/'status.json').write_text(json.dumps({'status': 'queued'}))
    reader = LogReader(arm)
    queued = reader.snapshot()
    assert queued['train']['map:loss'] == [[1, 1., 1.], [2, 2., 1.5]]
    assert queued['val']['map:render_l1'] == [[1, 1.], [2, 2.]]
    assert queued['status'] == {'status': 'queued'}
    assert queued['latest'] is None and queued['latest_validation'] is None
    assert queued['rows_read'] == 0 and queued['inherited_history']['rows_read'] == 2
    (arm/'train.jsonl').write_text(json.dumps({'stage': 'render', 'step': 1, 'total_step': 3, 'loss': 7.})+'\n')
    (arm/'validation.json').write_text(json.dumps({'history': [{'stage': 'render', 'step': 0, 'total_step': 2, 'render_l1': .1}]}))
    result = reader.snapshot()
    assert result['train_smooth']['render:loss'] == [[3, 7., 7.]]
    assert result['latest']['total_step'] == 3 and result['latest_validation']['stage'] == 'render'
    assert result['rows_read'] == 1
    assert result['val']['render:render_l1'] == [[2, .1]]
    assert (parent/'train.jsonl').read_text() == original
    assert reader.snapshot()['inherited_history']['rows_read'] == 2


def test_plateau_uses_ndr_and_separates_stages():
    history=[{'stage':'map','step':i*1000,'map_normal':1.,'map_diffuse':1.,'map_roughness':1.,'map_specular':i} for i in range(11)]
    assert plateau(history,'map')['status']=='plateau'
    assert plateau(history,'render')['status']=='observing'
    history.append({'stage':'map','step':11000,'map_normal':.9,'map_diffuse':.9,'map_roughness':.9})
    assert plateau(history,'map')['streak']==0
    history[-1]['map_normal']=float('nan')
    assert plateau(history,'map')['status']=='missing_or_invalid_metric'


def test_trainer_plateau_uses_recorded_ndrs_state_and_policy(tmp_path):
    history = [{'stage': 'map', 'map_normal': 1., 'map_diffuse': 1., 'map_roughness': 1.} for _ in range(30)]
    status = {'convergence': {
        'map': {'meaningful_best': .12, 'floor_bad_observations': 7, 'floor_reached': True},
        'render': {'meaningful_best': None, 'floor_bad_observations': 0, 'floor_reached': False}}}
    policy = {'floor_observations': 20, 'map_monitor': 'heldout.ndrs_sum', 'render_monitor': 'heldout.render_l1'}
    (tmp_path/'validation.json').write_text(json.dumps({'history': history}))
    (tmp_path/'status.json').write_text(json.dumps(status))
    (tmp_path/'execution_config.json').write_text(json.dumps({'policy': policy}))
    result = LogReader(tmp_path).snapshot()['plateau']
    assert result['map']['source'] == 'trainer'
    assert result['map']['metric_label'] == 'NDRS sum'
    assert result['map']['status'] == 'floor_wait'
    assert (result['map']['streak'], result['map']['target']) == (7, 20)
    assert result['render']['status'] == 'not_started'
    assert result['render']['metric_label'] == 'Render L1'
    # A meaningful improvement resets the trainer counter even if legacy diagnostics remain flat.
    status['convergence']['map'].update(meaningful_best=.119, floor_bad_observations=0)
    assert plateau(history, 'map', status, policy)['streak'] == 0
    policy['floor_observations'] = 8
    status['convergence']['map']['floor_bad_observations'] = 8
    assert plateau(history, 'map', status, policy)['status'] == 'plateau'
    assert plateau(history, 'map', status, policy)['target'] == 8


def test_trainer_plateau_does_not_invent_missing_policy():
    status = {'convergence': {'map': {'meaningful_best': .12, 'floor_bad_observations': 7, 'floor_reached': True}}}
    result = plateau([], 'map', status)
    assert result['source'] == 'trainer'
    assert result['status'] == 'missing_trainer_policy_or_state'
    assert result['target'] is None


def test_incremental_partial_and_rewritten_logs(tmp_path):
    path=tmp_path/'train.jsonl'
    first=json.dumps({'total_step':1,'stage':'map','loss':3.})+'\n'
    second=json.dumps({'total_step':2,'stage':'map','loss':1.})+'\n'
    path.write_text(first+second[:15],encoding='utf-8')
    reader=LogReader(tmp_path)
    assert reader.snapshot()['rows_read']==1
    with path.open('a',encoding='utf-8') as f:
        f.write(second[15:])
    state=reader.snapshot()
    assert state['rows_read']==2
    assert state['train']['map:loss'][-1]==[2,1.,2.]
    assert reader.snapshot()['rows_read']==2
    path.write_text(json.dumps({'total_step':1,'stage':'map','loss':7.})+'\n',encoding='utf-8')
    state=reader.snapshot()
    assert state['rows_read']==1
    assert state['train']['map:loss']==[[1,7.,7.]]


def test_val_points_and_missing_fields_are_not_fabricated(tmp_path):
    (tmp_path/'validation.json').write_text(json.dumps({'history':[{'stage':'map','step':1000,'map_normal':.3,'map_diffuse':.2,'map_roughness':.1}]}),encoding='utf-8')
    state=LogReader(tmp_path).snapshot()
    assert state['latest_validation']['step'] == 1000
    assert state['timestamps']['validation_file'].endswith('+00:00')
    assert state['timestamps']['train_log'] is None
    assert len(state['val']['map:ndr_mean'])==1
    assert abs(state['val']['map:ndr_mean'][0][1]-.2)<1e-9
    assert 'map:normal_degrees' not in state['val']
    assert not state['train']


def test_display_reduction_preserves_independent_extrema():
    from live_training_plot import display_points
    points = [[i, 0., 0.] for i in range(40)]
    points[2][1] = 10.
    points[6][2] = 20.
    raw = display_points(points, 1, limit=16)
    smooth = display_points(points, 2, limit=16)
    assert points[2] in raw
    assert points[6] in smooth
    assert points[6] not in raw
    assert raw[0] == smooth[0] == points[0]
    assert raw[-1] == smooth[-1] == points[-1]
    assert len(raw) <= 16 and len(smooth) <= 16
    assert len(points) == 40


def test_wrong_json_types_are_reported_without_breaking_snapshot(tmp_path):
    (tmp_path / 'train.jsonl').write_text('[]\n{"total_step":1,"stage":"map","loss":2}\n')
    reader = LogReader(tmp_path)
    for invalid in ([], {'history': {}}, {'history': [[]]}, {'history': [{'category': {'Fabric': []}}]}):
        (tmp_path / 'validation.json').write_text(json.dumps(invalid))
        result = reader.snapshot()
        assert result['invalid_train_rows'] == 1 and result['rows_read'] == 1
        assert result['validation_error'] and result['latest_validation'] is None


def test_ndrs_sum_is_preserved_separately_from_ndr_mean(tmp_path):
    row={"total_step":1,"stage":"map","map_aggregation":"ndrs_sum","map_loss":1.,"ndrs_sum":1.,"ndr_mean":.2,"specular_loss":.4}
    (tmp_path/"train.jsonl").write_text(json.dumps(row)+"\n")
    val={"stage":"map","total_step":0,"map_normal":.1,"map_diffuse":.2,"map_roughness":.3,"map_specular":.4,"ndrs_sum":1.,"category":{"Fabric":{"ndrs_sum":1.,"map_specular":.4}}}
    (tmp_path/"validation.json").write_text(json.dumps({"history":[val]}))
    state=LogReader(tmp_path).snapshot()
    assert state["train"]["map:ndrs_sum"]==[[1,1.,1.]]
    assert state["train"]["map:ndr_mean"]==[[1,.2,.2]]
    assert state["val"]["map:ndrs_sum"]==[[0,1.]]
    assert state["val"]["map:Fabric:map_specular"]==[[0,.4]]
