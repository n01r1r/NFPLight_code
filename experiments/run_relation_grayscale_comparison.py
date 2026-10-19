"""Append paired relation/NDRS inference to a frozen patch20 report (CPU FP32)."""
import argparse
import importlib.util
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from experiments.relation_grayscale_comparison import relation_numerator, weighted_grayscale_relation
from model.relation_geometry import RELATION_VERSION, spatial_far
from capture_processing.memory_refresh import _linear, prepare_memory_group
from capture_processing.report import _render_array, _save_prediction_component_previews, _stats
from capture_processing.difference_visualization import display_range, field_image, map_difference
from run_fabric_checkpoint_comparison import read, write, identity, hwc

METHODS = ('rgb_mean', 'weighted_grayscale')
COMPONENTS = {'normal': (0, 3), 'diffuse': (3, 6), 'roughness': (6, 7), 'specular': (7, 10)}
RELATIONS = ('numerator', 'relation_raw', 'relation_log_raw', 'relation', 'relation_log')
NOTE = ('RGB 평균: (|ΔR|+|ΔG|+|ΔB|)/3. 가중 grayscale: |0.299ΔR+0.587ΔG+0.114ΔB|. '
        'Δ=clip(Kθ×F×gain×Far,0,1)−Near, 선형 RGB에서 gamma 없이 계산합니다. '
        'grayscale 변환 후 절댓값이므로 채널 부호가 상쇄될 수 있습니다. '
        'gain은 두 방식 모두 중앙20×20 RGB 평균 near/far 비율입니다. '
        '입력·프레임·checkpoint·기하·FP32·decoder가 같고 feature18/19만 달라집니다. '
        '각 relation 단계의 표시 범위는 같은 sample의 두 방식·두 checkpoint 전체에서 공유합니다. '
        'N은 (XYZ+1)/2, D/R/S는 물리값 [0,1] 공통 표시이며 gamma off입니다. '
        'S는 원시 예측 head이며 renderer의 고정 F0=.04와 구분합니다. '
        '이 모델에서 D/S는 appearance 첫18채널만 사용하고 relation18/19는 N/R 분기에만 연결됩니다. '
        '따라서 appearance가 같으면 D/S 차이0은 구조상 예상되는 결과입니다. '
        '차이는 grayscale−RGB이고 실측 GT가 없어 정확도 개선을 뜻하지 않습니다.')


def scalar_stats(value):
    return {'mean': float(np.mean(value)), 'mean_abs': float(np.mean(np.abs(value))),
            'min': float(np.min(value)), 'max': float(np.max(value))}


def insert_section(path, section):
    page = path.read_text(encoding='utf8')
    assert 'id="relation-aggregation-comparison"' not in page, 'completed comparison must be preserved'
    assert '</main>' in page
    path.write_text(page.replace('</main>', section + '</main>', 1), encoding='utf8')


def run(report_root):
    report_root = report_root.resolve()
    output = report_root / 'relation_comparison'
    spec = importlib.util.spec_from_file_location('frozen_report_runner', report_root / 'run_real_comparison.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    for name in source.CAPTURE_GROUPS:
        baseline = read(report_root / name / 'manifest.json')
        if baseline.get('model_contract', {}).get('relation_version') != RELATION_VERSION:
            raise ValueError('baseline relation_version differs; regenerate baseline in a new report first')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    models, weights = source.load_models()
    protected = {str(p): identity(p) for p in source.SUITE.iterdir() if p.is_file()}
    protected.update({str(p): identity(p) for p in report_root.rglob('*') if p.is_file() and output not in p.parents and p.suffix != '.html'})
    contract = {'format': 'nfplight.relation-aggregation-comparison.v1', 'relation_version': RELATION_VERSION, 'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                'methods': {'rgb_mean': 'mean(abs(signed RGB), channels)',
                            'weighted_grayscale': 'abs(sum(signed RGB * [0.299,0.587,0.114], channels))'},
                'weights_domain': 'linear RGB; gamma off; these are declared grayscale weights, not linear-sRGB XYZ luminance coefficients',
                'gain': 'unchanged author_patch20 RGB mean ratio', 'checkpoints': weights,
                'precision': 'CPU FP32; AMP/TF32 off', 'changed_feature_channels': [18, 19],
                'display_population': 'all four method/checkpoint arms per sample', 'note': NOTE,
                'sha256_executed': False, 'npz_saved': False}
    write(output / 'experiment_plan.json', contract)
    groups = []
    sections = []
    for name in source.CAPTURE_GROUPS:
        dest = output / name
        dest.mkdir()
        old = read(report_root / name / 'manifest.json')
        chosen = old['selected_frames']['selected_indices']['near']
        assert chosen == old['selected_frames']['selected_indices']['far']
        g = prepare_memory_group(source.ARCHIVE / name, dest / 'preparation')
        assert read(g['output'] / 'registration.json') == read(report_root / name / 'preparation' / 'registration.json')
        inputs = []
        for side in ('near', 'far'):
            counts = np.mean(np.stack([g['counts'][side][i] for i in chosen]), axis=0, dtype=np.float32)
            values = _linear(counts, g['metadata'][side]['metadata'][0], g['manifest']['color_transforms'][side])
            inputs.append(values[-1])
        x = torch.from_numpy(np.concatenate(inputs, -1).transpose(2, 0, 1)[None].copy())
        saved = {}
        verification = {}
        gain = x[:, :3, 118:138, 118:138].mean() / x[:, 3:, 118:138, 118:138].mean().clamp(min=1e-5)
        assert float(gain) == old['gain_measurement']['gain']
        for arm, _ in source.ARMS:
            m = models[arm]
            features = {}
            for method in METHODS:
                context = weighted_grayscale_relation(m) if method == 'weighted_grayscale' else nullcontext()
                with context, torch.inference_mode():
                    f, _, _ = m.build_capture_features(x)
                    p = source.unit_prediction(m.net_g(f))
                    assert torch.equal(p, source.unit_prediction(m.net_g(f)))
                assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in (x, f, p))
                assert (p[:, :3].norm(dim=1) - 1).abs().max() < 2e-6
                features[method] = f
                signed = spatial_far(x[:, 3:] * gain, m.time_co_map, m.spatial_attenuation_factor).clamp(0, 1) - x[:, :3]
                numerator = relation_numerator(signed, method)
                ratio = numerator / m.coefficient.clamp(min=1e-5)
                log_ratio = ratio.clamp(min=1e-5).log()
                assert torch.equal(f[:, 18:19], (1 - m.ClipToOne(ratio)) * 2 - 1)
                assert torch.equal(f[:, 19:20], (1 - m.ClipToOne(log_ratio)) * 2 - 1)
                values = {'numerator': hwc(numerator), 'relation_raw': hwc(ratio),
                          'relation_log_raw': hwc(log_ratio), 'relation': hwc((f[:, 18:19] + 1) / 2),
                          'relation_log': hwc((f[:, 19:20] + 1) / 2), 'prediction': hwc(p)}
                arm_dir = dest / arm / method
                arm_dir.mkdir(parents=True)
                _save_prediction_component_previews(arm_dir, {'both': {'prediction': values['prediction']}})
                if method == 'rgb_mean':
                    previous = read(report_root / name / arm / 'manifest.json')['conditions']['both']['arrays']
                    for key in ('prediction', 'relation_raw', 'relation_log_raw', 'relation', 'relation_log'):
                        assert _stats(values[key]) == previous[key], f'baseline numerical drift: {name}/{arm}/{key}'
                    for component in COMPONENTS:
                        key = 'prediction_' + component + ('_rgb' if component != 'roughness' else '')
                        new_png = arm_dir / 'both/display' / (key + '.png')
                        old_png = report_root / name / arm / 'both/display' / (key + '.png')
                        assert np.array_equal(np.asarray(Image.open(new_png)), np.asarray(Image.open(old_png)))
                saved[(arm, method)] = values
                print('INFER', name, arm, method, flush=True)
            assert torch.equal(features['rgb_mean'][:, :18], features['weighted_grayscale'][:, :18])
            assert torch.equal(features['rgb_mean'][:, 20:], features['weighted_grayscale'][:, 20:])
            verification[arm] = {'repeat_exact': True, 'baseline_stats_and_png_exact': True,
                                 'unchanged_appearance_and_mask_exact': True, 'trace_matches_actual_features_exact': True}
        ranges = {}
        for key in RELATIONS:
            arrays = [v[key] for v in saved.values()]
            limits = [0., 1.] if key in ('relation', 'relation_log') else [float(min(a.min() for a in arrays)), float(max(a.max() for a in arrays))]
            ranges[key] = limits
            for (arm, method), values in saved.items():
                _render_array(values[key], 'scalar', dest / arm / method / (key + '.png'), tuple(limits), 'gray')
        metrics = {}
        difference_fields = {}
        for arm, _ in source.ARMS:
            before, after = saved[(arm, 'rgb_mean')], saved[(arm, 'weighted_grayscale')]
            metrics[arm] = {key: scalar_stats(after[key] - before[key]) for key in RELATIONS}
            for component, (start, end) in COMPONENTS.items():
                b, a = before['prediction'][..., start:end], after['prediction'][..., start:end]
                if component != 'normal':
                    b, a = (b + 1) / 2, (a + 1) / 2
                field = map_difference(b, a, component)
                difference_fields[(arm, component)] = field
                metrics[arm][component] = scalar_stats(field)
        delta_ranges = {}
        for component in COMPONENTS:
            scale = display_range([difference_fields[(a, component)] for a, _ in source.ARMS], signed=component == 'roughness')
            delta_ranges[component] = scale
            for arm, _ in source.ARMS:
                label = {'normal': 'Normal angle (deg)', 'diffuse': 'Diffuse RGB RMS', 'roughness': 'Roughness signed delta', 'specular': 'Predicted S RGB RMS'}[component]
                field_image(difference_fields[(arm, component)], scale, label).save(dest / arm / ('difference_' + component + '.png'))
        section = '<section id="relation-aggregation-comparison" class="results"><h2>RGB 평균 ↔ 가중 grayscale · relation 및 최종 N/D/R/S</h2><p>' + NOTE + '</p>'
        section += '<style>.aggregation-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;max-width:850px}.aggregation-pair img{width:100%;max-width:300px;height:auto}.aggregation-pair figure{margin:0;padding:8px}.aggregation-pair figcaption{font-size:13px}</style>'
        prefix = '../relation_comparison/' + name
        for arm, label in source.ARMS:
            section += '<h3>' + label + '</h3>'
            for key in (*RELATIONS, *COMPONENTS):
                limits = ranges.get(key, [0., 1.])
                section += '<h4>' + key + f' · 공통 표시 [{limits[0]:.7g}, {limits[1]:.7g}]</h4><div class="aggregation-pair">'
                for method in METHODS:
                    filename = key + '.png' if key in RELATIONS else 'both/display/prediction_' + key + ('_rgb' if key != 'roughness' else '') + '.png'
                    section += f'<figure><img src="{prefix}/{arm}/{method}/{filename}" alt="{arm} {method} {key}"><figcaption>{method}</figcaption></figure>'
                section += '</div>'
            section += '<h4>최종 맵 차이 · grayscale − RGB</h4><div class="aggregation-pair">'
            for component in COMPONENTS:
                section += f'<figure><img src="{prefix}/{arm}/difference_{component}.png" alt="{component} inference difference"><figcaption>{component} · 범례는 두 checkpoint 공통</figcaption></figure>'
            section += '</div><table><thead><tr><th>지표</th><th>평균</th><th>절댓값 평균</th><th>최소</th><th>최대</th></tr></thead><tbody>'
            for key, record in metrics[arm].items():
                section += '<tr><th>' + key + '</th>' + ''.join(f'<td>{record[k]:.7g}</td>' for k in ('mean', 'mean_abs', 'min', 'max')) + '</tr>'
            section += '</tbody></table>'
        section += f'<p><a href="{prefix}/comparison.json">비교 수치·범위·검증 기록</a></p></section>'
        record = {'group': name, 'gain': float(gain), 'selected_indices': chosen, 'relation_display_ranges': ranges,
                  'physical_difference_display_ranges': delta_ranges, 'metrics': metrics, 'verification': verification,
                  'metric_units': {'normal': 'degrees', 'diffuse': 'linear RGB reflectance RMS per pixel',
                                   'roughness': 'signed physical roughness delta', 'specular': 'predicted S linear RGB reflectance RMS per pixel',
                                   'relation': 'dimensionless normalized relation delta'}, 'note': NOTE}
        write(dest / 'comparison.json', record)
        sections.append((name, section))
        groups.append(record)
        assert all(identity(Path(p)) == v for p, v in g['protected'].items())
        g['counts'].clear()
        g['previews'].clear()
    assert all(identity(Path(p)) == v for p, v in protected.items()), 'pre-existing numerical assets changed'
    write(output / 'verification.json', {'status': 'passed', 'groups': groups, 'fresh_predictions': 20, 'repeat_predictions': 20,
          'baseline_reproduced_exact': True, 'source_and_existing_assets_size_mtime_unchanged': True,
          'sha256_executed': False, 'independent_review': 'pending'})
    for name, section in sections:
        insert_section(report_root / name / 'report.html', section)
    summary = '<section id="relation-aggregation-comparison"><h2>RGB 평균 / 가중 grayscale 추론 비교</h2><p>' + NOTE + '</p><ul>'
    for name, _ in sections:
        summary += f'<li><a href="{name}/report.html#relation-aggregation-comparison">{name} · 두 checkpoint 나란히 비교</a></li>'
    summary += '</ul><p><a href="relation_comparison/verification.json">비교 검증</a> · <a href="relation_comparison/experiment_plan.json">계산 계약</a></p></section>'
    insert_section(report_root / 'index.html', summary)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-root', type=Path, required=True)
    run(parser.parse_args().report_root)
