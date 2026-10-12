"""Numeric difference visualization: unclipped linear sRGB -> XYZ D65 -> Lab 2deg -> CIEDE2000.
No exposure, gamma or per-image normalization enters the metric (kL=kC=kH=1).
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import xyz2lab, deltaE_ciede2000
from matplotlib import colormaps


def linear_rgb_lab(rgb):
    values = np.asarray(rgb)
    if values.ndim < 2 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('expected finite linear RGB [...,3]')
    if np.any(values < 0):
        raise ValueError('negative linear RGB is outside the rendering colour contract')
    matrix = np.array([[.412453,.357580,.180423],[.212671,.715160,.072169],[.019334,.119193,.950227]], dtype=np.float64)
    return xyz2lab(values.astype(np.float64) @ matrix.T, illuminant='D65', observer='2')


def render_delta_e00(before, after):
    if np.shape(before) != np.shape(after):
        raise ValueError('render comparisons require identical shapes')
    result = deltaE_ciede2000(linear_rgb_lab(before), linear_rgb_lab(after))
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite CIEDE2000 difference')
    return result


def display_range(fields, *, signed=False):
    """Zero-anchored shared range; use P98 if isolated tails exceed it by 2x.
    All pixels remain in statistics. Zero difference retains a zero legend.
    """
    values = np.concatenate([np.asarray(field, dtype=np.float64).ravel() for field in fields])
    if not values.size or not np.isfinite(values).all():
        raise ValueError('difference range requires nonempty finite fields')
    if not signed and np.any(values < 0):
        raise ValueError('unsigned difference must be nonnegative')
    magnitude = np.abs(values)
    maximum = float(magnitude.max())
    percentile = float(np.percentile(magnitude,98))
    upper = percentile if percentile > 0 and percentile < maximum/2 else maximum
    return {'min':-upper if signed else 0.,'max':upper,'data_min':float(values.min()),
            'data_max':float(values.max()),'mean':float(values.mean()),'p98_abs':percentile,
            'policy':'P98 absolute' if upper != maximum else 'full extrema',
            'clipped_fraction':float(np.mean(magnitude > upper)),'signed':signed}


def field_image(values, scale, label, *, legend=True):
    """Scalar heatmap with embedded unit/range/clipping legend."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('heatmap requires finite [H,W] values')
    upper,signed = scale['max'],scale['signed']
    normalized = np.full(values.shape,.5 if signed else 0.) if upper == 0 else (.5+values/(2*upper) if signed else values/upper)
    cmap = colormaps['coolwarm' if signed else 'viridis']
    image = Image.fromarray(np.rint(cmap(np.clip(normalized,0,1))[...,:3]*255).astype('uint8'))
    if not legend:
        return image
    width,height = image.size
    image_width = width
    width = max(width,128)
    canvas = Image.new('RGB',(width,height+87),'white')
    canvas.paste(image,((width-image_width)//2,0))
    draw = ImageDraw.Draw(canvas)
    draw.text((3,height+2),label,fill='black')
    strip = np.tile(np.linspace(0,1,width)[None,:],(10,1))
    canvas.paste(Image.fromarray(np.rint(cmap(strip)[...,:3]*255).astype('uint8')),(0,height+20))
    ticks = [f"{scale['min']:.4g}", f"{0 if signed else upper/2:.4g}", f"{upper:.4g}"]
    font = ImageFont.load_default(size=10)
    while sum(draw.textlength(text,font=font) for text in ticks)+18 > width and font.size > 6:
        font = ImageFont.load_default(size=font.size-1)
    for text,x,anchor in zip(ticks,(3,width/2,width-3),('lt','mt','rt')):
        draw.text((x,height+32),text,fill='black',font=font,anchor=anchor)
    clipped = float(np.mean(np.abs(values)>upper))
    draw.text((3,height+48),f"{scale['policy']}; clipped {clipped:.2%}",fill='black')
    draw.text((3,height+64),f"data {values.min():.4g} .. {values.max():.4g}",fill='black')
    return canvas


def png_bytes(values,scale,label):
    output = io.BytesIO()
    field_image(values,scale,label).save(output,format='PNG')
    return output.getvalue()


def map_difference(before,after,component):
    """HWC physical maps: degrees, signed roughness, RGB reflectance RMS."""
    before,after = np.asarray(before),np.asarray(after)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('map comparisons require same-shaped finite physical maps')
    if component == 'normal':
        n0,n1 = np.linalg.norm(before,axis=-1,keepdims=True),np.linalg.norm(after,axis=-1,keepdims=True)
        if np.any(n0 == 0) or np.any(n1 == 0):
            raise ValueError('zero-length normal is invalid')
        a,b = before/n0,after/n1
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),(a*b).sum(-1)))
    if component == 'roughness':
        return (after-before)[...,0]
    if component in ('diffuse','specular'):
        return np.sqrt(np.mean(np.square(after.astype(np.float64)-before),axis=-1))
    raise ValueError(f'unsupported material component: {component}')


LABELS = {'normal':'Normal angle (deg)', 'diffuse':'Diffuse RGB RMS reflectance',
          'roughness':'Roughness signed delta', 'specular':'F0 RGB RMS reflectance'}


def saved_comparison(folder):
    """Refresh display assets from saved FP32 terminal maps/renders only."""
    import torch
    from capture_processing.rendering_loss_analysis import physical_maps
    folder = Path(folder)
    render_path, map_path = folder/'centered_render_values.npz', folder/'render_materials.npz'
    with np.load(render_path) as saved:
        renders = {key:saved[key].copy() for key in ('original','updated','near')}
    with np.load(map_path) as saved:
        raw_maps = {key:saved[key].copy() for key in ('original','updated')}
    all_arrays = {**renders,**{'map_'+key:value for key,value in raw_maps.items()}}
    for key,value in all_arrays.items():
        channels = 10 if key.startswith('map_') else 3
        if value.dtype != np.float32 or value.ndim != 4 or value.shape[:2] != (1,channels) or not np.isfinite(value).all():
            raise ValueError(f'{folder}: {key} requires finite FP32 [1,{channels},H,W]')
    maps = {key:{name:value[0].numpy().transpose(1,2,0) for name,value in physical_maps(torch.from_numpy(raw)).items()} for key,raw in raw_maps.items()}
    renders = {key:value[0].transpose(1,2,0) for key,value in renders.items()}
    for stage in ('original','updated'):
        for key,value in maps[stage].items():
            pixels = (value+1)/2 if key == 'normal' else value
            if pixels.shape[-1] == 1: pixels = np.repeat(pixels,3,axis=-1)
            Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype('uint8')).save(folder/f'difference_{stage}_{key}.png')
        Image.fromarray(np.rint(np.clip(renders[stage],0,1)*255).astype('uint8')).save(folder/f'difference_{stage}_render.png')
    fields = {key:map_difference(maps['original'][key],maps['updated'][key],key) for key in LABELS}
    fields.update(render=render_delta_e00(renders['original'],renders['updated']),
                  original_near=render_delta_e00(renders['near'],renders['original']),
                  updated_near=render_delta_e00(renders['near'],renders['updated']))
    scales = {key:display_range([value],signed=key=='roughness') for key,value in fields.items()}
    shared = display_range([fields['original_near'],fields['updated_near']])
    scales.update(original_near=shared,updated_near=shared)
    for key,value in fields.items():
        label = LABELS.get(key,'CIEDE2000 Delta E00')
        field_image(value,scales[key],label).save(folder/f'difference_{key}_legend.png')
    record = {'schema':'nfplight.physical-difference-display.v1','sources':[render_path.name,map_path.name],
              'precision':'saved arrays FP32, color conversion FP64 CPU',
              'render_metric':'CIEDE2000; unclipped linear sRGB -> XYZ D65 -> CIELAB 2deg; kL=kC=kH=1',
              'map_metrics':LABELS,'display_ranges':scales,'statistics_include_all_pixels':True,
              'range_population':'each map/pair independently; original/updated versus I_N share one range',
              'no_inference_optimization_gamma_exposure_or_sha256':True}
    (folder/'difference_display.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return record


def comparison_html(folder,prefix,record):
    """Embed newly generated physical differences with legends in every panel."""
    prefix = prefix.rstrip('/')
    blocks = []
    for stage,title in (('original','Original prediction'),('updated','Updated prediction')):
        panels = []
        for key in LABELS:
            panels.append(f'<figure><figcaption>{key}</figcaption><img src="{prefix}/difference_{stage}_{key}.png" alt="{title} {key}"></figure>')
        panels.append(f'<figure><figcaption>Near re-render</figcaption><img src="{prefix}/difference_{stage}_render.png" alt="{title} centered render"></figure>')
        blocks.append(f'<h3>{title}</h3><div class="physical-difference-grid">'+''.join(panels)+'</div>')
    panels = []
    for key,label in {**LABELS,'render':'Rendering Delta E00'}.items():
        panels.append(f'<figure><figcaption>{label}</figcaption><img src="{prefix}/difference_{key}_legend.png" alt="{label}; embedded range legend"></figure>')
    blocks.append('<h3>Difference · updated versus original</h3><div class="physical-difference-grid">'+''.join(panels)+'</div>')
    panels = [f'<figure><figcaption>{key.title()} render versus I_N · Delta E00</figcaption><img src="{prefix}/difference_{key}_near_legend.png" alt="CIEDE2000 render versus measurement"></figure>' for key in ('original','updated')]
    blocks.append('<h3>Re-rendering versus measured near</h3><div class="physical-difference-grid">'+''.join(panels)+'</div>')
    blocks.append('<p>차이 색상과 범례: normal 각도(°), diffuse/F0 선형 RGB 반사율 RMS(부호 없음), roughness 물리값 부호 차이, 렌더링 CIEDE2000 ΔE00. RGB RMS는 채널 방향을 표시하지 않습니다. 각 영상의 실제 min/max와 표시 클리핑 비율을 범례에 기록합니다. 극단값이 P98의 두 배를 넘으면 P98 표시 상한을 사용합니다. 지표와 기존 MSE는 모든 픽셀을 유지합니다. 원본·갱신 렌더링 대 I_N은 같은 범위입니다. 다른 비교의 색 강도는 범례 범위를 함께 확인하세요.</p><p>ΔE00: clip/gamma/노출 없이 선형 sRGB→XYZ D65→Lab(2°), kL=kC=kH=1. 음수 또는 비유한 RGB는 오류입니다. 절대 RGB 영상은 gamma off이며 표시만 [0,1] clip합니다. RGB가 1보다 큰 경우 D65 white Y=1의 HDR 확장값이며 디스플레이 JND 판정을 뜻하지 않습니다. 기존 loss·맵·FP32 수치 파일은 변경하지 않습니다.</p>')
    return '<style>.physical-difference-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.physical-difference-grid figure{margin:0}.physical-difference-grid img{width:100%;height:auto}.physical-difference-grid figcaption{font-size:12px}@media(max-width:800px){.physical-difference-grid{grid-template-columns:repeat(2,1fr)}}</style>'+''.join(blocks)
