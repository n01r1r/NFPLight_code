"""Read-only exposure/distribution audit; DeepMaterials is reference only."""
from collections import Counter
import json,random
from pathlib import Path
import numpy as np
from PIL import Image

BANK=Path('C:/MatSynth_PRIMA_Fabric')
DEEP=Path('D:/DeepMaterialsData_cropped')
OUT=Path('artifacts/fabric_leather_data_audit_20261007')


def js(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a/a.sum();b=b/b.sum();m=(a+b)/2
    return float(.5*sum(np.sum(x[x>0]*np.log2(x[x>0]/m[x>0])) for x in (a,b)))


def summarize(samples):
    values={'roughness':[],'normal_tilt_deg':[],'diffuse_encoded':[],'diffuse_gamma22':[]}
    for n,d,r in samples:
        n=n.astype(np.float32)/127.5-1
        length=np.linalg.norm(n,axis=-1,keepdims=True)
        if np.any(length<1e-5): raise ValueError('zero source normal')
        n=n/length
        values['normal_tilt_deg'].append(np.degrees(np.arccos(np.clip(n[...,2],-1,1))).ravel())
        values['roughness'].append((r.astype(np.float32)/255).ravel())
        encoded=d.astype(np.float32)/255
        values['diffuse_encoded'].append(encoded.ravel())
        values['diffuse_gamma22'].append((encoded**2.2).ravel())
    result={}
    for key,chunks in values.items():
        a=np.concatenate(chunks)
        upper=180 if key=='normal_tilt_deg' else 1
        hist=np.histogram(a,bins=64,range=(0,upper))[0]
        result[key]={'mean':float(a.mean()),'std':float(a.std()),'p05_p50_p95':np.quantile(a,[.05,.5,.95]).tolist(),
                     'histogram':hist.tolist(),'pixel_value_count':len(a),'hist_range':[0,upper]}
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=json.loads((DEEP/'split_manifest.json').read_text())
    inventory={}
    for label,paths in split['splits'].items():
        recipes=Counter();assets=Counter()
        for path in paths:
            parts=Path(path).name.split(';')
            if len(parts)>1:
                recipes[parts[1]]+=1;assets.update(parts[1].split('X'))
        inventory[label]={'images':len(paths),'recipe_names':len(recipes),'constituent_asset_names':len(assets)}
    rows=[json.loads(s) for s in (BANK/'manifests/fabric_leather_split_manifest.jsonl').read_text().splitlines() if s.strip()]
    train=[r for r in rows if r['split']=='train']
    sample_count=64
    rng=random.Random(7)
    deep_paths=rng.sample(split['splits']['train'],sample_count)
    deep_samples=[];deep_sizes=Counter();identities={'DeepMaterials':[]}
    for raw in deep_paths:
        filename=Path(raw).name
        arrays=[]
        for component in ['Normal','Diffuse','Roughness']:
            path=DEEP/'train'/component/filename
            with Image.open(path) as im:
                w,h=im.size
                if component=='Normal': deep_sizes[f'{w}x{h}']+=1
                if min(w,h)<256: raise ValueError(f'small map: {path}')
                im=im.crop(((w-256)//2,(h-256)//2,(w+256)//2,(h+256)//2))
                if component=='Roughness': im=im.convert('L')
                a=np.asarray(im.resize((64,64),Image.Resampling.NEAREST)).copy()
            arrays.append(a)
        deep_samples.append(tuple(arrays));identities['DeepMaterials'].append(filename)
    distributions={'DeepMaterials':summarize(deep_samples)}
    for category in ['Fabric','Leather']:
        eligible=sorted([r for r in train if r['category']==category],key=lambda r:r['material_id'])
        chosen=random.Random(7).sample(eligible,min(sample_count,len(eligible)))
        samples=[];identities[category]=[]
        for row in chosen:
            with np.load(row['shard_path'],allow_pickle=False) as z:
                for crop in [0,4,8,12]:
                    arrays=[z[k][crop,::4,::4].copy() for k in ['normal','diffuse','roughness']]
                    samples.append(tuple(arrays))
            identities[category].append(row['material_id'])
        distributions[category]=summarize(samples)
    divergence={category:{key:js(distributions['DeepMaterials'][key]['histogram'],distributions[category][key]['histogram'])
                          for key in distributions['DeepMaterials']} for category in ['Fabric','Leather']}
    crop_count=len(train)*16;reference=inventory['train']['images']
    exposure={}
    for b in [1,4,8]:
        exposures=500000*b
        exposure[str(b)]={'effective_batch':b,'total_presentations':exposures,
                         'subset_mean_presentations_per_stored_crop':exposures/crop_count,
                         'deep_mean_presentations_per_stored_image':exposures/reference,
                         'subset_mean_presentations_per_material':exposures/len(train),
                         'subset_update_count_matching_deep_per_image_repetition':500000*crop_count/reference}
    report={'deep_reference_only':True,'deep_inventory':inventory,'subset_train_materials':len(train),
            'subset_train_counts_by_category':dict(Counter(r['category'] for r in train)),
            'subset_train_sources_by_category':{c:dict(Counter(r['source'] for r in train if r['category']==c)) for c in ['Fabric','Leather']},
            'subset_train_groups':len({r['group_id'] for r in train}),'subset_stored_train_crops':crop_count,
            'stored_crop_repetition_ratio_same_updates_batch':reference/crop_count,
            'exposure_scenarios_400k_plus_100k':exposure,'sample_identities':identities,
            'distribution_sampling':{'seed':7,'deep_images':sample_count,'deep_source_sizes':dict(deep_sizes),'subset_materials_per_category':sample_count,
                                     'subset_crops_per_material':4,'pixel_grid':'64x64 nearest subsample',
                                     'deep_crop':'central256','subset_mask':'none; four-map-only contract',
                                     'diffuse_transfer':'both encoded and gamma22 hypotheses; physical transfer not established'},
            'distributions':distributions,'js_divergence_bits':divergence,
            'limitations':['descriptive marginal distributions, not model performance or physical GT',
                           'DeepMaterials crops and MatSynth materials are different statistical units',
                           'native256 MatSynth crops and DeepMaterials prepared288 crops have uncalibrated physical scale',
                           'fixed 40 training-light directions cover a bounded subset, not general light/view diversity',
                           'matching repeated crop exposure does not match semantic diversity or optimizer trajectory',
                           'DeepMaterials random256 crops within288 would add up to1089 translated windows per prepared image; actual NFPLight training augmentation unverified'],
            'sha256_executed':False}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'train_materials':len(train),'stored_crops':crop_count,'repetition_ratio':reference/crop_count,'js':divergence}),flush=True)

if __name__=='__main__': main()
