"""Recover original binary decisions from losslessly verified archive constraints.

This does not retrain models, change thresholds, or optimize predictions. Most
float16 values determine their original float32 threshold decisions uniquely.
For the few intervals crossing the threshold, enumerate EVERY possible binary
assignment and retain only assignments matching all five archived image metrics.
Store a scalar only when all compatible assignments agree; otherwise retain the
full candidate values and their bounds. Never choose a solution using the
corrected metric, and never overwrite source inputs.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, itertools, json, sys
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from surface_metrics import surface_distance

METRICS=('dice','iou','hd95','assd','biou')


def old_metrics(pr,gt):
    intersection=(pr&gt).sum()
    d={'dice':float((2*intersection+1e-7)/(pr.sum()+gt.sum()+1e-7)),
       'iou':float((intersection+1e-7)/((pr|gt).sum()+1e-7))}
    pb=pr&~binary_erosion(pr);gb=gt&~binary_erosion(gt)
    if pr.any() and gt.any():
        distances=np.r_[distance_transform_edt(~gt)[pb],distance_transform_edt(~pr)[gb]]
        d.update(hd95=float(np.percentile(distances,95)),assd=float(distances.mean()))
    else:
        d.update(hd95=float('nan'),assd=float('nan'))
    pd=distance_transform_edt(~pb);gd=distance_transform_edt(~gb)
    d['biou']=float(((pb&(gd<=2)).sum()+(gb&(pd<=2)).sum())/(pb.sum()+gb.sum()+1e-7))
    return d


def equal(a,b):
    return bool((np.isnan(a) and np.isnan(b)) or abs(a-b)<=1e-10)


def recover(q,gt,t,archived):
    if q.dtype!=np.float16:
        raise ValueError('Expected archived float16 probabilities.')
    # Conservative intervals include both midpoint ties. Over-inclusion can only
    # create an unresolved case, never rule out the original binary prediction.
    value=q.astype(np.float32)
    lower=(np.nextafter(q,np.float16(-np.inf)).astype(np.float32)+value)/2
    upper=(np.nextafter(q,np.float16(np.inf)).astype(np.float32)+value)/2
    threshold=np.float32(t)
    ambiguous=np.flatnonzero((lower<threshold)&(upper>=threshold))
    fixed=lower>=threshold
    if len(ambiguous)>16:
        raise RuntimeError('Too many ambiguous pixels; no heuristic recovery allowed.')
    compatible=[]
    for bits in itertools.product((False,True),repeat=len(ambiguous)):
        pred=fixed.copy();pred.ravel()[ambiguous]=bits
        intersection=(pred&gt).sum()
        dice=float((2*intersection+1e-7)/(pred.sum()+gt.sum()+1e-7))
        iou=float((intersection+1e-7)/((pred|gt).sum()+1e-7))
        if not equal(dice,archived['dice']) or not equal(iou,archived['iou']):
            continue
        old=old_metrics(pred,gt)
        if all(equal(old[k],archived[k]) for k in METRICS):
            compatible.append((pred,surface_distance(pred,gt)))
    status='unique_mask' if len(compatible)==1 else 'unresolved'
    if len(compatible)>1 and all(all(equal(a,b) for a,b in zip(c[1],compatible[0][1])) for c in compatible[1:]):
        status='unique_metrics_multiple_masks'
    return ambiguous,compatible,status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output exists; use a new directory to preserve the audit.')
    args.output.mkdir(parents=True)
    cases=[];unresolved=[];summaries=[]
    for path in sorted(args.source.glob('*/result.json')):
        name=path.parent.name
        result=json.loads(path.read_text(encoding='utf-8'))
        rows=json.loads((path.parent/'per_image.json').read_text(encoding='utf-8'))
        updated=copy.deepcopy(rows)
        destination=args.output/name;destination.mkdir()
        binaries=destination/'predictions_binary';binaries.mkdir()
        for i,row in enumerate(rows):
            file=path.parent/'predictions'/(row['id']+'.npz')
            with np.load(file,allow_pickle=False) as z:
                q=z['prob'];gt=z['gt']>.5
            if not equal(float(gt.mean()),row['area']):
                raise RuntimeError('Ground-truth area mismatch.')
            legacy={**row,**row.get('legacy_surface_metrics',{})}
            ambiguous,compatible,status=recover(q,gt,result['threshold'],legacy)
            updated[i]['legacy_surface_metrics']={k:legacy[k] for k in ('hd95','assd')}
            record=dict(run=name,id=row['id'],ambiguous_pixels=len(ambiguous),
                        compatible_masks=len(compatible),status=status,
                        source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            cases.append(record)
            if not compatible:
                raise RuntimeError(f'No compatible binary mask: {name}/{row["id"]}')
            if status=='unresolved':
                record['status']='bounded_metrics_multiple_masks'
                unresolved.append(record)
                updated[i]['surface_metric_candidates']=[list(c[1]) for c in compatible]
            for j,key in enumerate(('hd95','assd')):
                values=[c[1][j] for c in compatible]
                if all(equal(v,values[0]) for v in values):
                    updated[i][key]=values[0]
                else:
                    updated[i][key]=None
                    updated[i].setdefault('surface_metric_bounds',{})[key]=[min(values),max(values)]
            if status=='unique_mask':
                np.savez_compressed(binaries/(row['id']+'.npz'),
                                    pred=compatible[0][0].astype(np.uint8),gt=gt.astype(np.uint8))
        old_summary=result.get('legacy_surface_metrics',{k:result['test'][k] for k in ['hd95','assd']})
        result['legacy_surface_metrics']=old_summary
        for key in ['hd95','assd']:
            if any(r[key] is None for r in updated):
                bounds=[[r[key],r[key]] if r[key] is not None else r['surface_metric_bounds'][key] for r in updated]
                result['test'][key]=None
                result.setdefault('surface_metric_bounds',{})[key]=np.nanmean(bounds,axis=0).tolist()
            else:
                result['test'][key]=float(np.nanmean([r[key] for r in updated]))
        result['surface_metric_definition']='Boundary-to-boundary; pooled bidirectional HD95 and point-weighted ASSD; empty-mask exclusions unchanged.'
        result['surface_metric_provenance']='Exhaustive float16 rounding-interval enumeration constrained by all five archived per-image metrics. Non-unique corrected values are null with bounds and all candidate values retained; see surface_metric_recovery_audit.csv.'
        (destination/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        (destination/'per_image.json').write_text(json.dumps(updated,indent=2),encoding='utf-8')
        summaries.append(dict(run=name,old_hd95=old_summary['hd95'],new_hd95=result['test']['hd95'],
                              old_assd=old_summary['assd'],new_assd=result['test']['assd']))
        print(name,'HD95',round(result['test']['hd95'],6),'ASSD',result['test']['assd'] if result['test']['assd'] is not None else result['surface_metric_bounds']['assd'],flush=True)
    with (args.output/'surface_metric_recovery_audit.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    summary={'runs_total':len(list(args.source.glob('*/result.json'))),
             'runs_processed':len(summaries),'images_total':len(cases),
             'unique_masks':sum(x['status']=='unique_mask' for x in cases),
             'unique_metrics_multiple_masks':sum(x['status']=='unique_metrics_multiple_masks' for x in cases),
             'bounded_cases':unresolved,'max_ambiguous_pixels':max(x['ambiguous_pixels'] for x in cases),
             'acceptance_tolerance':1e-10,'runs':summaries}
    (args.output/'recovery_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2),flush=True)


if __name__=='__main__':main()
