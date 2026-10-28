"""Validate corrected metrics against archived inputs, binaries and constraints."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from recompute_archived_surfaces import old_metrics, recover, equal, METRICS
from surface_metrics import surface_distance

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; A=ROOT/'metric_correction'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
audit={(x['run'],x['id']):x for x in csv.DictReader((A/'surface_metric_recovery_audit.csv').open(encoding='utf-8'))}
runs=list(R.glob('*/result.json'))
images=unique=bounded=excluded=0
for path in runs:
    result=load(path); rows=load(path.parent/'per_image.json')
    assert len(rows)==149
    for row in rows:
        images+=1
        source=path.parent/'predictions'/(row['id']+'.npz')
        record=audit[(path.parent.name,row['id'])]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
        legacy={**row,**row['legacy_surface_metrics']}
        with np.load(source,allow_pickle=False) as z:
            gt=z['gt']>.5
            binary=path.parent/'predictions_binary'/(row['id']+'.npz')
            if binary.exists():
                unique+=1
                with np.load(binary,allow_pickle=False) as mask:
                    pred=mask['pred'].astype(bool)
                    assert np.array_equal(mask['gt']>.5,gt)
                # Check the recovered mask against every archived metric.
                previous=old_metrics(pred,gt)
                assert all(equal(previous[k],legacy[k]) for k in METRICS)
                corrected=surface_distance(pred,gt)
                assert all(equal(row[k],v) for k,v in zip(('hd95','assd'),corrected))
                if np.isnan(corrected[0]): excluded+=1
            else:
                bounded+=1
                _,compatible,status=recover(z['prob'],gt,result['threshold'],legacy)
                assert len(compatible)==int(record['compatible_masks'])>1
                np.testing.assert_allclose(row['surface_metric_candidates'],[list(c[1]) for c in compatible],rtol=0,atol=1e-12)
                for j,key in enumerate(('hd95','assd')):
                    values=[c[1][j] for c in compatible]
                    if row[key] is None:
                        np.testing.assert_allclose(row['surface_metric_bounds'][key],[min(values),max(values)],rtol=0,atol=1e-12)
                    else:
                        assert all(equal(row[key],v) for v in values)
        assert float(gt.mean())==row['area']
        for key in ('hd95','assd'):
            lower=row[key] if row[key] is not None else row['surface_metric_bounds'][key][0]
            assert np.isnan(lower)==np.isnan(legacy[key])
            if np.isfinite(lower): assert lower>=legacy[key]-1e-10
    for key in ('dice','iou','biou','hd95','assd'):
        if result['test'][key] is None:
            bounds=[[r[key],r[key]] if r[key] is not None else r['surface_metric_bounds'][key] for r in rows]
            np.testing.assert_allclose(result['surface_metric_bounds'][key],np.nanmean(bounds,axis=0),rtol=0,atol=1e-12)
        else:
            assert equal(result['test'][key],float(np.nanmean([r[key] for r in rows])))
    for key in ('hd95','assd'):
        assert equal(result['legacy_surface_metrics'][key],float(np.nanmean([r['legacy_surface_metrics'][key] for r in rows])))
assert (len(runs),images,unique,bounded)==(41,6109,6107,2)

diag=load(R/'revision_diagnostics.json')
missing_modes=0
for row in diag['mechanism']:
    original=load(R/f'kvasir_bcskg_s{row["seed"]}_r0.05_b0.1'/'result.json')
    assert row['selected']==original['test']
    assert load(R/f'kvasir_bcskg_s{row["seed"]}_mechanism.json')==row
    for mode in ('base_only','graph_only'):
        entry=row[mode]
        if entry['test']['hd95'] is None:
            assert entry['test']['assd'] is None
            missing_modes+=1
        else:
            assert entry['fusion']==row['selected_fusion'] and entry['threshold']==row['selected_threshold']
assert missing_modes==5

report={'status':'passed','training_runs':len(runs),'archived_prediction_files_verified':images,
        'unique_binary_masks_verified':unique,'bounded_assd_cases_verified':bounded,
        'empty_mask_cases':excluded,'frozen_mode_runs_without_surface_metrics':missing_modes,
        'checks':['Original probability SHA256s','Archived overlap and legacy surface metrics',
                  'Corrected boundary distances and monotonicity','Per-run means and bounds',
                  'Frozen-mode missing-value handling']}
(A/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
