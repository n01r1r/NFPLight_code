import json
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_fabric_leather_subset import crop_shard


def test_native_crops_are_aligned_repeatable_and_lossless(tmp_path):
    src=tmp_path/'sample'
    src.mkdir()
    yy,xx=np.indices((272,272))
    rgb=np.stack((xx%256,yy%256,(xx+yy)%256),axis=-1).astype(np.uint8)
    for name in ('diffuse','normal','specular'):
        Image.fromarray(rgb).save(src/(name+'.png'))
    Image.fromarray((xx%256).astype(np.uint8)).save(src/'roughness.png')
    (src/'metadata.json').write_text(json.dumps({'metadata':{'maps':[]}}))
    a,b=tmp_path/'a.npz',tmp_path/'b.npz'
    crop_shard(src,a);crop_shard(src,b)
    with np.load(a) as first,np.load(b) as second:
        for key in first.files:
            assert np.array_equal(first[key],second[key])
        assert 'valid_mask' not in first.files
        assert set(first.files)=={'normal','diffuse','roughness','specular','origins','reference_size','resolution'}
        assert len({tuple(p) for p in first['origins']})==16
        for i,(y,x) in enumerate(first['origins']):
            assert np.array_equal(first['diffuse'][i],rgb[y:y+256,x:x+256])
            assert np.array_equal(first['roughness'][i],rgb[y:y+256,x:x+256,0])
        assert np.array_equal(first['specular'],first['diffuse'])
    Image.fromarray(np.zeros((544,544),np.uint8)).save(src/'opacity.png')
    rejected=tmp_path/'rejected.npz'
    assert crop_shard(src,rejected) is not None
    with np.load(rejected) as third,np.load(a) as first:
        for key in first.files: assert np.array_equal(first[key],third[key])


def test_js_divergence_limits():
    from audit_fabric_leather_data import js
    assert abs(js([1,0],[1,0]))<1e-12
    assert abs(js([1,0],[0,1])-1)<1e-12
    assert abs(js([1,2],[3,4])-js([3,4],[1,2]))<1e-12


def test_uint16_roughness_and_rgba_ignore_alpha(tmp_path):
    src=tmp_path/'rgba';src.mkdir()
    rgba=np.zeros((272,272,4),np.uint8);rgba[:,:,:3]=80
    Image.fromarray(rgba).save(src/'diffuse.png')
    for name in ['normal','specular']: Image.fromarray(rgba[:,:,:3]).save(src/(name+'.png'))
    Image.fromarray(np.full((272,272),32768,np.uint16)).save(src/'roughness.png')
    (src/'metadata.json').write_text('{}')
    dst=tmp_path/'map.npz';crop_shard(src,dst)
    with np.load(dst) as z:
        assert z['diffuse'].shape==(16,256,256,3)
        assert (z['diffuse']==80).all()
        assert (z['roughness']==128).all()
