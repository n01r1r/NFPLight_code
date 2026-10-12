"""Extend the frozen Fabric bank with Leather; source PNGs and Fabric stay intact.

Native uint8 maps, four material maps only, 16x256 crops, group split/seed7.
No SHA-256 is calculated. Original specular is retained for later fixed-F0 training.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from urllib.parse import urlsplit
import numpy as np
from PIL import Image

BANK=Path('C:/MatSynth_PRIMA_Fabric')
SOURCE=Path('D:/MatSynth_preprocessed')
MAN=BANK/'manifests'
MAPS={'diffuse':3,'normal':3,'roughness':1,'specular':3}
SIZE=256
COUNT=16

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def crop_shard(src,dst):
    arrays={}
    for name,channels in MAPS.items():
        with Image.open(src/(name+'.png')) as im: a=np.asarray(im).copy()
        if channels==3 and a.ndim==3 and a.shape[2]==4: a=a[:,:,:3].copy()
        if channels==1 and a.dtype==np.uint16: a=np.rint(a.astype(np.float64)/257).astype(np.uint8)
        if a.dtype!=np.uint8 or a.ndim!=(2 if channels==1 else 3) or (channels==3 and a.shape[2]!=3):
            raise ValueError(f'invalid {name} source: {src}')
        arrays[name]=a
    shapes={a.shape[:2] for a in arrays.values()}
    if len(shapes)!=1: raise ValueError(f'map shape mismatch: {src}')
    h,w=shapes.pop()
    if min(h,w)<SIZE: raise ValueError(f'too small: {src}')
    rng=random.Random(f'7:Leather:{src.name}')
    selected=rng.sample(range((h-SIZE+1)*(w-SIZE+1)),COUNT)
    origins=np.array([divmod(i,w-SIZE+1) for i in selected],dtype=np.int32)
    payload={k:np.stack([a[y:y+SIZE,x:x+SIZE] for y,x in origins]) for k,a in arrays.items()}
    payload.update(origins=origins,reference_size=np.array(w,np.int32),resolution=np.array(SIZE,np.int32))
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**payload)
    with np.load(tmp,allow_pickle=False) as z:
        for k,a in payload.items():
            if not np.array_equal(z[k],a): raise ValueError(f'roundtrip failed: {dst}/{k}')
    tmp.replace(dst)
    return {'format':'four_maps_native_v1','source_size_hw':[h,w],'origins_yx':origins.tolist(),'size_bytes':dst.stat().st_size,
            'source_file_stats':{p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in [src/'metadata.json',*(src/(k+'.png') for k in MAPS)]}}

def main():
    combined=MAN/'fabric_leather_split_manifest.jsonl'
    if combined.exists(): raise FileExistsError(combined)
    old=MAN/'fabric_split_manifest.jsonl'
    old_bytes=old.read_bytes()
    fabric=[dict(json.loads(line),category='Fabric') for line in old_bytes.decode().splitlines() if line.strip()]
    preserved={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in BANK.rglob('*') if p.is_file() and 'Leather' not in p.parts and not p.name.startswith(('leather_', 'fabric_leather_'))}
    rows=[];excluded=[]
    for split in ['train','test']:
        folder=SOURCE/split/'Leather'
        if not folder.is_dir(): raise FileNotFoundError(folder)
        for src in sorted(p for p in folder.iterdir() if p.is_dir()):
            meta=json.loads((src/'metadata.json').read_text())
            if meta['category']!='Leather' or meta['split']!=split: raise ValueError(f'identity mismatch: {src}')
            dst=BANK/'source_shards'/split/'Leather'/(src.name+'.npz')
            provenance=dst.with_suffix('.json')
            if dst.exists() and provenance.exists() and json.loads(provenance.read_text()).get('format')=='four_maps_native_v1':
                details=json.loads(provenance.read_text())
                if dst.stat().st_size!=details['size_bytes']: raise ValueError(f'shard size changed: {dst}')
                for filename,expected in details['source_file_stats'].items():
                    stat=(src/filename).stat()
                    if [stat.st_size,stat.st_mtime_ns]!=expected: raise ValueError(f'source changed: {src}/{filename}')
            else:
                details=crop_shard(src,dst)
                if details is None:
                    excluded.append({'category':'Leather','material_id':src.name,'official_split':split,'reason':'fewer than 16 distinct 95%-opaque crops'})
                    continue
                save(provenance,details)
            source=meta['metadata']['source']
            key=source+':'+(re.sub(r'^\d+_','',src.name) if source=='deschaintre_2020' else src.name)
            rows.append({'material_id':src.name,'category':'Leather','source':source,'source_dir':str(src),
                         'official_split':split,'shard_path':str(dst),'group_id':key,
                         'link':meta['metadata'].get('link','').rstrip('/'),
                         'identity_method':'size_mtime_array_roundtrip_no_sha256'})
            if len(rows)%10==0:
                save(MAN/'leather_preparation_progress.json',{'prepared':len(rows),'excluded':excluded})
                print('Prepared',len(rows),'Leather materials',flush=True)
    links={};groups=defaultdict(list)
    for r in rows:
        if r['link'] and urlsplit(r['link']).path.strip('/'):
            r['group_id']=links.setdefault((r['source'],r['link']),r['group_id'])
        groups[r['group_id']].append(r)
    tests={r['group_id'] for r in rows if r['official_split']=='test'}
    candidates=sorted(g for g in groups if g not in tests)
    random.Random(7).shuffle(candidates)
    val=set();n=0;target=round(.2*sum(len(groups[g]) for g in candidates))
    for g in candidates[:-1]:
        if n>=target: break
        val.add(g);n+=len(groups[g])
    kept=[]
    for r in rows:
        if r['official_split']=='train' and r['group_id'] in tests:
            excluded.append({'material_id':r['material_id'],'reason':'shares_group_with_official_test'})
            continue
        r['split']='test' if r['official_split']=='test' else ('val' if r['group_id'] in val else 'train')
        kept.append(r)
    all_rows=fabric+kept
    membership=defaultdict(set)
    for r in all_rows: membership[r['group_id']].add(r['split'])
    if any(len(v)!=1 for v in membership.values()): raise ValueError('cross-split group leakage')
    if len({(r['category'],r['material_id']) for r in all_rows})!=len(all_rows): raise ValueError('duplicate material identity')
    if old.read_bytes()!=old_bytes: raise ValueError('Fabric manifest changed')
    for p,stats in preserved.items():
        now=Path(p).stat()
        if [now.st_size,now.st_mtime_ns]!=stats: raise ValueError(f'existing bank file changed: {p}')
    summary={'schema_version':1,'status':'complete','categories':['Fabric','Leather'],
             'counts_by_category':{c:dict(Counter(r['split'] for r in all_rows if r['category']==c)) for c in ['Fabric','Leather']},
             'counts':dict(Counter(r['split'] for r in all_rows)),
             'source_root':str(SOURCE),'destination':str(BANK),'excluded':excluded,
             'rules':{'crop_size':SIZE,'crops_per_material':COUNT,'seed':7,'leather_val_fraction':.2,
                      'used_maps':list(MAPS),'opacity_used':False,'crop_sampling':'all native windows, per-material string seed',
                      'grouping':'source links and repeated Deschaintre descriptors; Fabric frozen',
                      'specular_storage':'source uint8 preserved; trainer applies fixed F0'},
             'checks':{'existing_files_preserved':len(preserved),'group_splits_disjoint':True,'array_roundtrip':True,'sha256_executed':False},
             'limitations':['visual duplicates across distinct source links not audited','historical Fabric crop RNG not reconstructed']}
    save(MAN/'fabric_leather_split_summary.json',summary)
    temp=combined.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows),encoding='utf-8')
    temp.replace(combined)
    save(MAN/'leather_preparation_progress.json',{'status':'complete','prepared':len(rows),'excluded':excluded})
    print(json.dumps(summary['counts_by_category']),flush=True)

if __name__=='__main__': main()
