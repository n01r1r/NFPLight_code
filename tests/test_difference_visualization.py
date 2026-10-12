import numpy as np
import pytest
from skimage.color import deltaE_ciede2000
from capture_processing.difference_visualization import render_delta_e00,linear_rgb_lab,display_range,field_image,map_difference


def test_ciede2000_reference_and_linear_contract():
    assert deltaE_ciede2000(np.array([50.,2.6772,-79.7751]),np.array([50.,0.,-82.7485])) == pytest.approx(2.0425,abs=5e-5)
    rgb = np.array([[[0.,0.,0.],[.5,.5,.5],[1.,1.,1.]]],np.float32)
    assert np.max(render_delta_e00(rgb,rgb)) == 0
    assert linear_rgb_lab(rgb)[0,1,0] == pytest.approx(76.0693,abs=.001)
    assert linear_rgb_lab(np.ones((1,1,3))*2)[0,0,0] > 100
    np.testing.assert_allclose(render_delta_e00(rgb,rgb*.6),render_delta_e00(rgb*.6,rgb))
    with pytest.raises(ValueError,match='negative'):
        render_delta_e00(-np.ones((1,1,3)),rgb[:,:1])
    with pytest.raises(ValueError,match='finite'):
        render_delta_e00(np.full((1,1,3),np.nan),rgb[:,:1])


def test_adaptive_range_preserves_extrema_and_legend():
    values = np.full((20,20),.02)
    values[0,0] = 8
    scale = display_range([values])
    assert scale['max'] == .02 and scale['data_max'] == 8
    assert scale['clipped_fraction'] == 1/400
    assert field_image(values,scale,'Delta E00').height == values.shape[0]+87
    zero = display_range([np.zeros((2,2))])
    assert zero['max'] == zero['clipped_fraction'] == 0
    assert field_image(np.zeros((2,2)),zero,'Delta E00')


def test_physical_map_differences():
    n0,n1 = np.array([[[0.,0.,2.]]]),np.array([[[3.,0.,0.]]])
    assert map_difference(n0,n1,'normal').item() == 90
    assert map_difference(np.ones((1,1,1))*.7,np.ones((1,1,1))*.4,'roughness').item() == pytest.approx(-.3)
    assert map_difference(np.zeros((1,1,3)),np.array([[[.3,0.,0.]]]),'diffuse').item() == pytest.approx(.3/np.sqrt(3))
    with pytest.raises(ValueError,match='zero-length'):
        map_difference(n0*0,n1,'normal')



def test_browser_delta_e_matches_python_and_scalar_semantics():
    import json
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    rng = np.random.default_rng(7)
    before,after = rng.uniform(0,2,(40,3)),rng.uniform(0,2,(40,3))
    script = Path('capture_processing/difference_visualization.js').read_text(encoding='utf-8')
    script += "\nconst pairs="+json.dumps(list(zip(before.tolist(),after.tolist())))+"; console.log(JSON.stringify(pairs.map(([a,b])=>ciede2000(linearLab(a,0),linearLab(b,0)))));"
    result = subprocess.run([node,'-e',script],capture_output=True,text=True,check=True)
    np.testing.assert_allclose(json.loads(result.stdout),render_delta_e00(before,after),rtol=3e-6,atol=3e-6)



@pytest.mark.parametrize('upper,signed,width',[(7.195,False,256),(.03,True,256),(1.234e-10,True,64),(0.,False,128)])
def test_heatbar_ticks_anchor_to_left_center_right_without_overlap(monkeypatch,upper,signed,width):
    from PIL import ImageDraw
    calls=[]
    original=ImageDraw.ImageDraw.text
    def observe(self,xy,text,*args,**kwargs):
        if kwargs.get('anchor'):
            calls.append((xy,text,kwargs['anchor'],kwargs['font']))
        return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',observe)
    values=np.zeros((20,width))
    scale={'min':-upper if signed else 0.,'max':upper,'signed':signed,'policy':'full extrema'}
    image=field_image(values,scale,'test')
    assert [call[2] for call in calls] == ['lt','mt','rt']
    assert [call[0][0] for call in calls] == [3,image.width/2,image.width-3]
    draw=ImageDraw.Draw(image)
    boxes=[draw.textbbox(xy,text,font=font,anchor=anchor) for xy,text,anchor,font in calls]
    assert boxes[0][0]>=0 and boxes[-1][2]<=image.width
    assert boxes[0][2]<boxes[1][0] and boxes[1][2]<boxes[2][0]



def test_explorer_export_tick_anchors_match_bar_endpoints():
    import re,json,shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    template=Path('cue_explorer_template.html').read_text(encoding='utf-8')
    code=re.search(r"const ticks=\[signed\?fmt\(-scale\).*?ctx.textAlign='left';",template,re.S).group()
    for signed in (False,True):
        harness="const scale=.03,signed="+str(signed).lower()+";const cv={width:512},fmt=x=>String(x),calls=[],ctx={fillText(...args){calls.push([this.textAlign,...args]);}};"+code+"console.log(JSON.stringify(calls));"
        result=subprocess.run([node,'-e',harness],capture_output=True,text=True,check=True)
        calls=json.loads(result.stdout)
        assert [call[0] for call in calls]==['left','center','right']
        assert [call[2] for call in calls]==[12,256,500]
        assert [call[3] for call in calls]==[603]*3
        assert [call[1] for call in calls]==(['-0.03','0','0.03'] if signed else ['0','0.015','0.03'])
        assert all(call[4]<512/3 for call in calls)


def test_saved_physical_maps_preserve_decode_and_fp32_boundary():
    import torch
    from capture_processing.rendering_loss_analysis import physical_maps
    raw = torch.zeros(1, 10, 2, 2, dtype=torch.float32)
    raw[:, 2] = 2
    raw[:, 6] = -1
    maps = physical_maps(raw)
    assert torch.equal(maps['normal'][:, 2], torch.ones(1, 2, 2))
    assert torch.all(maps['diffuse'] == .5)
    assert torch.all(maps['roughness'] == .05)
    assert torch.all(maps['specular'] == .5)
    with pytest.raises(TypeError, match='FP32'):
        physical_maps(raw.double())
    invalid = raw.clone(); invalid[:, :3] = 0
    with pytest.raises(ValueError, match='zero normal'):
        physical_maps(invalid)
    invalid = raw.clone(); invalid[:, 3] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        physical_maps(invalid)
