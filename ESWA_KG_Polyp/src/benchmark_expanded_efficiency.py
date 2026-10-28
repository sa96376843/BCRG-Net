"""Sequential, warmed CPU latency audit for all seven controlled networks."""
import json, time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, RESULTS
from extended_evaluation import load_run

torch.set_num_threads(2)
SEEDS = [13, 29, 47]
def tag(method, seed):
    if method == 'unetwide':
        return f'kvasir_unet_s{seed}_r0.05_b0.1_base10'
    return f'kvasir_{method}_s{seed}_r0.05_b0.1'

payload = []
for method in ['unet','attention','attention_original','unetpp','deeplab','unetwide','bcskg']:
    result, model = load_run(tag(method, 13))
    model.eval()
    x = torch.rand(1, 3, 64, 64)
    times = []
    with torch.no_grad():
        for _ in range(5):
            model(x)
        for _ in range(30):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter()-start)*1000)
    payload.append({'method':method, 'parameters':result['params'],
                    'latency_ms_mean':float(np.mean(times)),
                    'latency_ms_sd':float(np.std(times,ddof=1)),
                    'warmup_passes':5, 'timed_passes':30})
    print(method, payload[-1], flush=True)
(RESULTS/'expanded_efficiency_benchmark.json').write_text(json.dumps(payload,indent=2))
