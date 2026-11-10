"""Optional isolated CUDA preview; never used as a quality evaluation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch

from colab_tools import relative_path
from nfplight import cli
from nfplight.checkpoint import build_model, decode, read_config
from nfplight.pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required = True)
    parser.add_argument('--out', required = True)
    parser.add_argument('--config', required = True)
    parser.add_argument('--training-config', default = 'configs/training_noquant.json')
    parser.add_argument('--runtime', default = 'configs/runtime_accum4.json')
    args = parser.parse_args()
    relative_path(args.manifest)
    for path in (args.config, args.training_config, args.runtime):
        relative_path(path)
    output = relative_path(args.out)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise ValueError('CUDA GPU preview required')
    profile = read_config(args.config)
    if profile['features'].get('synthetic_quantization', 'none') != 'none':
        raise ValueError('unquantized experiment profile required')
    runtime = cli.read_json(args.runtime)
    cli.configure_runtime(runtime, 'cuda')
    execution = cli.execution_from_config(SimpleNamespace(manifest = args.manifest,
        training_config = args.training_config, smoke_updates = None), profile, runtime)
    torch.manual_seed(19)
    torch.cuda.manual_seed_all(19)
    pipeline = TrainingPipeline(execution, profile, runtime, device = 'cuda')
    with torch.device('cuda'):
        network = build_model(profile).eval()
    shapes, hooks = [], []
    names = ('intro_albedo', 'intro_specular', 'encoders_albedo.', 'encoders_specular.',
             'middle_blks_albedo', 'middle_blks_specular', 'decoder_albedo.fusions.',
             'decoder_specular.fusions.', 'albedoMap', 'specularMap')
    names += ('decoders_albedo.', 'decoders_specular.', 'ups_albedo.', 'ups_specular.')
    def capture(name):
        def hook(module, inputs, result):
            shapes.append({'module': name, 'input': list(inputs[0].shape), 'output': list(result.shape)})
        return hook
    for name, module in network.named_modules():
        if name in names or any(name.startswith(prefix) and '.' not in name[len(prefix):]
                                for prefix in names if prefix.endswith('.')):
            hooks.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            target, mask, features = pipeline.prepare(pipeline.val.crop(0, 0)[None])
            pair = pipeline.renderer.observations(target, 4., 12.) * mask
            _, trace = pipeline.features(pair, purpose = 'synthetic', mask = mask)
            raw = network(features)
            decoded = decode(raw, profile['decoder'])
        if list(features.shape) != [1, 21, 256, 256] or list(decoded.shape) != [1, 10, 256, 256]:
            raise ValueError('pipeline shape contract mismatch')
    finally:
        for hook in hooks:
            hook.remove()
    def image(value):
        return value[0].detach().cpu().permute(1, 2, 0).numpy().squeeze()
    panels = [('Near linear RGB', pair[:, :3]), ('Far linear RGB', pair[:, 3:]),
              ('Aligned far (gain9)', trace['aligned_far']), ('Relation', trace['relation']),
              ('Log relation', trace['log_relation']), ('GT normal (XYZ+1)/2', (target[:, :3] + 1) / 2),
              ('GT diffuse linear', (target[:, 3:6] + 1) / 2), ('GT roughness', (target[:, 6:7] + 1) / 2)]
    fig, axes = plt.subplots(2, 4, figsize = (15, 8), layout = 'constrained')
    for ax, (title, value) in zip(axes.flat, panels):
        ax.imshow(image(value), vmin = 0, vmax = 1, cmap = 'gray', interpolation = 'nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Pipeline trace only — source FP32 arrays, fixed [0,1], no display gamma')
    output.mkdir(parents = True)
    fig.savefig(output / 'features.png', dpi = 140)
    (output / 'shapes.json').write_text(json.dumps({'parameters': sum(p.numel() for p in network.parameters()),
        'features': list(features.shape), 'raw_output': list(raw.shape), 'decoded_output': list(decoded.shape),
        'forward_modules': shapes, 'weights': 'fresh random, not a trained quality result',
        'official_test_payloads_read': 0}, indent = 2), encoding = 'utf-8')


if __name__ == '__main__':
    main()
