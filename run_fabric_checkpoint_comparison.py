"""Fresh FP32 Fabric comparison; reuses current capture and HTML helpers."""
import argparse
import copy
import html
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np
import torch
from capture_processing.memory_refresh import _linear, prepare_memory_group
from capture_processing.report import (_FORMULAS, _PREDICTION_COMPONENTS,
                                       _case_image, _case_results_html,
                                       _case_row, _display_configs, _make_html,
                                       _make_range, _save_display_previews,
                                       _save_prediction_component_previews,
                                       _stats)
from model.nfplight_matsynth_capture_model import MatSynthCaptureModel
from model.author_real_capture_adapter import GAIN_PATCH
from run_fabric_capture import CAPTURE_GROUPS

PARENT = Path('checkpoints/fabric_amp_l1map400k_render100k_20260915')
RESTART = Path(
    'checkpoints/fabric_amp_l1map400k_render100k_azimuth9_polar45_topview_20260918_r2'
)
WEIGHTS = [
    ('map400k', '400k · map 종료', PARENT / 'stage_a.pth', 400000, ('render', 0,
                                                                  400000)),
    ('best449k', '449k · 예전 best_render', RESTART / 'best_render.pth', 449000,
     ('render', 49000, 149000)),
    ('final500k', '500k · 최종', RESTART / 'last.step-00200000.complete.pt',
     500000, ('complete', 100000, 200000))
]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, v):
    Path(p).write_text(json.dumps(v,
                                  ensure_ascii=False,
                                  indent=2,
                                  allow_nan=False),
                       encoding='utf-8')


def identity(p):
    return (p.stat().st_size, p.stat().st_mtime_ns)


def hwc(t):
    return t.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()


def load_models(gain_mode):
    models, records = {}, {}
    for key, label, path, cumulative, expected in WEIGHTS:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload['state']
        assert tuple(state[k]
                     for k in ('stage', 'step', 'total_step')) == expected
        assert payload['feature_version'] == 'raw_calibrated_v2'
        assert all(
            torch.is_tensor(v) and v.dtype == torch.float32
            and torch.isfinite(v).all() for v in payload['params'].values())
        m = MatSynthCaptureModel(
            SimpleNamespace(image_size=256,
                            precision='float32',
                            capture_precision=True,
                            device='cuda',
                            input_mode='linear_rgb',
                            capture_gain_mode=gain_mode,
                            loadpath_network_g=str(path)))
        m.net_g.eval()
        models[key] = m
        records[key] = {
            'label': label,
            'path': str(path.resolve()),
            'cumulative_updates': cumulative,
            'logged_state': dict(zip(('stage', 'step', 'total_step'),
                                     expected)),
            'feature_version': m.feature_version,
            'capture_gain_mode': gain_mode,
            'normal_head': m.normal_head,
            'training_amp': payload['contract']['settings']['amp'],
            'identity_size_mtime': identity(path)
        }
        del payload
    return models, records


def trace(m, x, f, p):
    n, far = x.chunk(2, 1)
    if m.args.capture_gain_mode == 'author_patch20':
        gain = n[:, :, GAIN_PATCH[0], GAIN_PATCH[1]].mean() / far[:, :, GAIN_PATCH[0], GAIN_PATCH[1]].mean().clamp(min=1e-5)
    else:
        gain = n.new_tensor((m.far_distance / m.near_distance)**2)
    gained = far * gain
    timed = (gained * m.time_co_map).clamp(0, 1)
    signed = timed - n
    numerator = signed.abs().mean(1, keepdim=True)
    denominator = m.coefficient.clamp(min=1e-5)
    ratio = numerator / denominator
    decoded = (f + 1) / 2
    return {
        k: hwc(v)
        for k, v in {
            'near_input_clipped': n,
            'far_input_clipped': far,
            'far_gain_map': gain.expand_as(m.coefficient),
            'far_gain_scalar': gain.reshape(1, 1, 1, 1),
            'far_gained_clipped': gained.clamp(0, 1),
            'far_time_clipped': timed,
            'signed_diff': signed,
            'abs_diff': signed.abs(),
            'mean_abs_diff': numerator,
            'coefficient_raw': m.coefficient_raw,
            'coefficient': m.coefficient,
            'denominator': denominator,
            'relation_raw': ratio,
            'relation_log_raw': ratio.clamp(min=1e-5).log(),
            'relation': decoded[:, 18:19],
            'relation_log': decoded[:, 19:20],
            'log_near': decoded[:, 9:12],
            'valid_mask': decoded[:, 20:21],
            'time_co_map': m.time_co_map,
            'features': f,
            'prediction': p
        }.items()
    }


def report(out, manifest, arrays, label):
    out.mkdir(parents=True, exist_ok=True)
    keys = [
        f'{k}_c{i}' for k, v in arrays.items()
        if v.shape[-1] > 3 and k != 'prediction' for i in range(v.shape[-1])
    ]
    keys += [k for k, v in arrays.items() if v.shape[-1] <= 3]
    stats = {k: _stats(v) for k, v in arrays.items()}
    stats.update({
        f'prediction_c{i}': _stats(arrays['prediction'][..., i:i + 1])
        for i in range(10)
    })
    manifest = copy.deepcopy(manifest)
    manifest['conditions'] = {
        'both': {
            'formula': '(D-B)/(W-B)',
            'arrays': stats
        }
    }
    ranges = {
        k: list(_make_range(k, {'both': arrays}, {}, manifest))
        for k in keys
    }
    _save_display_previews(out, ranges, manifest['conditions'], keys,
                           {'both': arrays}, manifest)
    _save_prediction_component_previews(out, {'both': arrays})
    configs = _display_configs(keys, ranges, {'both': arrays}, {})
    formulas = [
        tuple(s.replace('FP32 NPZ', 'FP32 메모리') for s in r) for r in _FORMULAS
    ]
    patch_gain = manifest['model_contract']['capture_gain_mode'] == 'author_patch20'
    if patch_gain:
        formulas = [
            (title,
             'g=mean(N[:,:,118:138,118:138])/max(mean(F[:,:,118:138,118:138]),1e-5)',
             '256×256 입력의 중앙 20×20 RGB 평균',
             '저자 실측 코드 L102의 영역 평균을 적용합니다.', key)
            if key == 'far_gain_map' else (title, lhs, rhs, meaning, key)
            for title, lhs, rhs, meaning, key in formulas
        ]
    gain_note = ('저자 실측 코드의 중앙 20×20 RGB 평균 gain. Fabric 21채널과 checkpoint 기하는 유지하며 학습의 고정 gain 계약과는 다른 입력 조건입니다.' if patch_gain else 'Fabric 21채널 raw_calibrated_v2: gain=(10/2.414)².')
    page = _make_html(
        manifest,
        keys,
        ranges,
        configs, {}, [],
        formula_records=formulas,
        notation_note=
        gain_note
    )
    page = page.replace('NPZ', '메모리 배열(저장 없음)')
    start = page.index('<p class="run-strip">Ω = Ω_c:')
    end = page.index('</p>', start) + 4
    page = page[:start] + '<p class="run-strip">' + gain_note + '</p>' + page[end:]
    page = page.replace(
        '<h1>Fabric capture · ',
        '<h1>Fabric checkpoint · ' + html.escape(label) + ' · ', 1)
    write(out / 'manifest.json', manifest)
    write(out / 'verification.json', manifest['verification'])
    (out / 'report.html').write_text(page, encoding='utf-8')
    return page, manifest, configs


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    protected = {p: identity(p) for _, _, p, *_ in WEIGHTS}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    plan = {
        'scope':
        'User authorized all three custom Fabric weights, separately from original-author reports.',
        'weights': [{
            'label': label,
            'path': str(p),
            'cumulative_updates': step
        } for _, label, p, step, _ in WEIGHTS],
        'input':
        'Preserved continuous-AHD FP32 camera RGB -> current common-plane warp/crop -> reviewed frames -> (D-B)/(W-B) -> WB/matrix once -> clip [0,1].',
        'invariants': [
            'No PNG input, integer quantization, gamma, denoiser, AMP, TF32 or SHA-256.',
            'Same FP32 input and 21-channel features for all weights.',
            'Keep checkpoint 21-channel layout, geometry and decoder; record approved gain override.',
            'Preserve original checkpoints and reports.'
        ],
        'acceptance': [
            'Actual registration review', 'Finite FP32 tensors',
            'Repeat inference equality', 'Input/feature equality',
            'Source/checkpoint size/mtime unchanged',
            'HTML asset and browser checks'
        ],
        'limitations': [
            'No measured SVBRDF GT; map differences are not accuracy.',
            'Training used AMP; this inference uses FP32.',
            'Reprocess preserved FP32 camera RGB; no fresh DNG demosaic.',
            'PNG/JSON/HTML only; no NPZ saved.'
        ]
    }
    plan['capture_gain_mode'] = args.gain_mode
    plan['gain_contract'] = {
        'patch': [118, 138, 118, 138],
        'aggregation': 'RGB and spatial mean, per sample',
        'denominator_floor': 1e-5,
        'source_code': 'https://github.com/CGLiWang/NFPLight_code/blob/main/model/nfplight_real_model.py#L102',
        'paper': 'https://cgliwang.github.io/NFPLight/pdf/NFPLight.pdf, Eq. (4)',
        'checkpoint_geometry_preserved': True
    }
    if args.gain_mode == 'author_patch20':
        plan['limitations'].append('Central 20x20 gain replaces training fixed gain by user instruction; not full original-author reproduction.')
    write(out / 'experiment_plan.json', plan)
    groups = [
        prepare_memory_group(args.archive / name, out / name / 'preparation')
        for name in CAPTURE_GROUPS
    ]
    for g in groups:
        assert read(g['output'] / 'registration.json') == read(
            args.current / g['archive'].name /
            'refresh_20261006/registration.json'), 'registration changed'
    if args.review_from is not None:
        for g in groups:
            prior = args.review_from / g['archive'].name / 'preparation'
            assert read(g['output'] / 'registration.json') == read(prior / 'registration.json'), 'review reuse requires identical registration'
            review = read(prior / 'root_review.json')
            assert review['status'] == 'reviewed' and isinstance(review['texture_registration_passed'], bool)
            review['reused_from'] = str((prior / 'root_review.json').resolve())
            review['reuse_basis'] = 'New FP32 registration records exactly equal previously reviewed records; gain change follows registration.'
            write(g['output'] / 'root_review.json', review)
    write(out / 'review_ready.json', {
        'registration_matches_current': True,
        'groups': list(CAPTURE_GROUPS)
    })
    print('REVIEW_READY', flush=True)
    deadline = time.monotonic() + 1200
    while any(
            read(g['output'] / 'root_review.json')['status'] != 'reviewed'
            for g in groups):
        if time.monotonic() > deadline:
            raise TimeoutError('visual review required')
        time.sleep(1)
    models, weight_records = load_models(args.gain_mode)
    results = []
    for g in groups:
        name = g['archive'].name
        review = read(g['output'] / 'root_review.json')
        assert isinstance(review['texture_registration_passed'], bool)
        chosen = list(
            range(5)
        ) if not g['failures'] and review['texture_registration_passed'] else [
            0
        ]
        arrays = {}
        for side in ('near', 'far'):
            counts = np.mean(np.stack([g['counts'][side][i] for i in chosen]),
                             axis=0,
                             dtype=np.float32)
            values = _linear(counts, g['metadata'][side]['metadata'][0],
                             g['manifest']['color_transforms'][side])
            arrays.update({
                f'{side}_{k}': v
                for k, v in zip(('compensated', 'whitebalanced',
                                 'input_unclipped', 'input_clipped'), values)
            })
        x = torch.from_numpy(
            np.concatenate(
                [arrays['near_input_clipped'], arrays['far_input_clipped']],
                -1).transpose(2, 0, 1)[None].copy()).cuda()
        base = {
            k: copy.deepcopy(g['manifest'][k])
            for k in ('source', 'capture_metadata', 'color_transforms')
        }
        base.update(
            group=name,
            format='nfplight.fabric-checkpoint-comparison.v1',
            selected_frames={
                'averaged': len(chosen) == 5,
                'selected_indices': {
                    s: chosen
                    for s in ('near', 'far')
                },
                'averaging_gate': {
                    'quantitative_passed': not g['failures'],
                    'reasons': g['failures']
                }
            },
            frame_average={
                'enabled':
                len(chosen) == 5,
                'near_count':
                len(chosen),
                'far_count':
                len(chosen),
                'fallback_reason':
                g['failures'],
                'texture_review_status':
                'passed' if review['texture_registration_passed'] else 'failed'
            },
            root_visual_review=review,
            source_arrays={
                'read_only': True,
                'provenance': str(g['archive'].resolve())
            },
            model_geometry={
                'near_distance': 2.414,
                'far_distance': 10,
                'capture_distances_cm': [10, 30]
            },
            model_contract={
                'feature_version': 'raw_calibrated_v2',
                'capture_gain_mode': args.gain_mode,
                'normal_head': 'xyz',
                'estimator_input_channels': 21
            },
            precision={
                'processing': 'float32',
                'inference': 'float32',
                'AMP': False,
                'TF32': False
            },
            denoiser={'used': False},
            numeric_storage='memory only; PNG/JSON/HTML outputs',
            created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            environment={
                'python': platform.python_version(),
                'torch': torch.__version__,
                'numpy': np.__version__,
                'gpu': torch.cuda.get_device_name(0),
                'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version()
            },
            limitations=plan['limitations'],
            report={})
        rows = []
        records = {}
        reference_features = None
        reference_prediction = None
        for key, label, *_ in WEIGHTS:
            m = models[key]
            with torch.inference_mode():
                m.feed_data({'inputs': x})
                m.test()
                p = m.pred_svbrdf.clone()
                f, _, _ = m.build_features(x, gain_mode=args.gain_mode)
                m.test()
                assert torch.equal(p, m.pred_svbrdf), 'repeat mismatch'
            if reference_features is None:
                reference_features = f.clone()
                reference_prediction = p.clone()
            assert torch.equal(f, reference_features), 'feature mismatch'
            assert p.shape == (1, 10, 256, 256)
            assert all(t.dtype == torch.float32 and torch.isfinite(t).all()
                       for t in (x, f, p))
            current_arrays = dict(arrays, **trace(m, x, f, p))
            gain = current_arrays['far_gain_scalar'].item()
            assert np.allclose(current_arrays['far_gained_clipped'], np.clip(current_arrays['far_input_clipped'] * gain, 0, 1), rtol=2e-6, atol=1e-7)
            assert torch.allclose((f[:, 6:9] + 1) / 2, (x[:, 3:] * gain).clamp(0, 1), rtol=2e-6, atol=1e-7)
            if args.gain_mode == 'author_patch20':
                nmean = x[:, :3, GAIN_PATCH[0], GAIN_PATCH[1]].mean().item()
                fmean = x[:, 3:, GAIN_PATCH[0], GAIN_PATCH[1]].mean().item()
                base['gain_measurement'] = dict(mode=args.gain_mode, patch=[118,138,118,138], near_patch_rgb_mean=nmean, far_patch_rgb_mean=fmean, gain=gain, gained_far_patch_rgb_mean=fmean * gain, patch_mean_residual=fmean * gain - nmean, denominator_floor_active=fmean < 1e-5)
            manifest = dict(base,
                            checkpoint=weight_records[key],
                            verification={
                                'finite_fp32': True,
                                'repeat_inference_equal': True,
                                'same_input_and_features': True,
                                'npz_saved': False,
                                'sha256_executed': False
                            })
            page, saved, configs = report(out / name / key, manifest,
                                          current_arrays, label)
            if key == 'map400k':
                first_page, first_manifest, first_configs = page, saved, configs
            records[key] = dict(weight_records[key],
                                mean_abs_raw_channel_difference_vs_400k=(
                                    p - reference_prediction).abs().mean(
                                        (0, 2, 3)).cpu().tolist(),
                                prediction_stats=_stats(
                                    current_arrays['prediction']),
                                verification=manifest['verification'])
            tiles = [
                _case_image(
                    f'{key}/both/display/{component}.png',
                    title,
                    f'{label} {title}',
                    gamma_source=f'{key}/both/display_gamma22/{component}.png'
                    if gamma else None)
                for component, title, _, _, _, gamma in _PREDICTION_COMPONENTS
            ]
            rows.append(_case_row(label, tiles))
            print('INFER_COMPLETE ' + name + ' ' + key, flush=True)
        old = _case_results_html(['both'],
                                 first_manifest['conditions'],
                                 first_configs,
                                 None,
                                 author_real=False,
                                 model_label='')
        tiles = [
            _case_image(f'map400k/both/display/{k}.png',
                        label,
                        label,
                        gamma_source=f'map400k/both/display_gamma22/{k}.png'
                        if k in {'near_input_clipped', 'far_input_clipped', 'far_gained_clipped', 'far_time_clipped'} else None)
            for k, label in [(
                'near_input_clipped',
                'Near'), ('far_input_clipped',
                          'Far'), ('far_gained_clipped', 'K_L I_F'), ('far_time_clipped', 'K_L K_θ I_F'), ('relation',
                                   'Relation'), ('relation_log',
                                                 'Log relation')]
        ]
        combined = '<section class="results"><h2>Results · 동일 입력 / 세 weight</h2>' + _case_row(
            '공통 입력', tiles) + ''.join(rows) + '</section>'
        first_page = first_page.replace(old, combined)
        first_page = first_page.replace(
            "mode+(gamma?'/display_gamma22':'/display')",
            "'map400k/'+mode+(gamma?'/display_gamma22':'/display')")
        notice = '<p class="run-strip">2026-10-08 새 FP32 재추론 · 동일 교정 입력 · 감마 기본 off. 아래 세 행을 비교하세요. 단계 inspector의 prediction은 400k입니다.</p><p>'
        notice += ' · '.join(
            f'<a href="{k}/report.html">{html.escape(label)} 상세</a>'
            for k, label, *_ in WEIGHTS)
        notice += '</p><p>500k = map 400k + render 100k. best_render = map 400k + render 49k. 재시작 로그 total_step(200k/149k)은 누적 학습량과 구분합니다. 이전 예측 결과를 재사용하지 않았습니다.</p>'
        if args.gain_mode == 'author_patch20':
            notice += '<p>중앙 20×20 RGB 평균으로 K_L을 계산합니다. K=K_L K_θ이며 두 보정 영상을 공통 입력 행에서 비교합니다. 중앙 평균 일치는 전체 픽셀 일치를 뜻하지 않습니다.</p>'
        first_page = first_page.replace('<main>', '<main>' + notice, 1)
        first_page = first_page.replace('Fabric checkpoint · 400k · map 종료 · ',
                                        'Fabric · 400k / 449k / 500k · ')
        (out / name / 'report.html').write_text(first_page, encoding='utf-8')
        write(out / name / 'manifest.json',
              dict(first_manifest, comparison_weights=records))
        write(out / name / 'verification.json', first_manifest['verification'])
        write(
            out / name / 'comparison.json', {
                'group': name,
                'weights': records,
                'selected_indices': chosen,
                'interpretation':
                'Raw map differences only; no accuracy ranking.'
            })
        results.append({
            'group': name,
            'selected_indices': chosen,
            'weights': records
        })
        assert all(
            identity(Path(p)) == old_identity
            for p, old_identity in g['protected'].items()), 'source changed'
        g['counts'].clear()
        g['previews'].clear()
    assert all(identity(p) == v
               for p, v in protected.items()), 'checkpoint changed'
    assert not list(out.rglob('*.npz'))
    write(
        out / 'verification.json', {
            'status': 'passed',
            'groups': len(results),
            'fresh_predictions': len(results) * 3,
            'source_checkpoint_size_mtime_unchanged': True,
            'sha256_executed': False,
            'npz_saved': False,
            'results': results
        })
    links = ''.join(
        f'<li><a href="{r["group"]}/report.html">{r["group"]}</a> · ' +
        ('5장 평균' if len(r['selected_indices']) == 5 else 'near_00/far_00') +
        '</li>' for r in results)
    style = first_page.split('<style>', 1)[1].split('</style>', 1)[0]
    (out / 'index.html').write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fabric 400k · 449k · 500k</title><style>'
        + style +
        '</style><main><h1>Fabric · 교정 입력 FP32 재추론</h1><p>400k 종료 / 449k best_render / 누적 500k 최종. 입력 양자화·감마·denoiser 없음. 기존 보고서의 단계 선택·맵 행·표시 감마 유지.</p><ul>'
        + links +
        '</ul><p><a href="verification.json">수치 검증</a> · <a href="experiment_plan.json">실행 계약</a></p></main></html>',
        encoding='utf-8')
    print('COMPLETE ' + str(out / 'index.html'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review-from', type=Path, help='Reuse review only after exact registration equality')
    parser.add_argument('--gain-mode', choices=('checkpoint', 'author_patch20'), default='author_patch20')
    parser.add_argument(
        '--archive',
        type=Path,
        default=Path('artifacts/fabric_capture_20261002_paper_equations'))
    parser.add_argument(
        '--current',
        type=Path,
        default=Path(
            'artifacts/fabric_capture_20261002_aligned_paper_equations'))
    run(parser.parse_args())
