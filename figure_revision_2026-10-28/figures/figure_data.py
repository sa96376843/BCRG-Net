"""Shared, read-only inputs for the BCRG-Net empirical figure refresh."""
from pathlib import Path
import json
import sys
import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parent / 'ESWA_KG_Polyp'
RESULTS = PROJECT / 'results'
DATA = PROJECT / 'data'
OUTPUT = WORK / 'experimental_redraw'
OUTPUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT / 'src'))

SEEDS = (13, 29, 47)
METHODS = ('unet', 'attention', 'attention_original', 'unetpp', 'deeplab', 'unetwide', 'bcskg')
NAME = {'unet':'U-Net', 'attention':'Skip-attention U-Net',
        'attention_original':'Attention U-Net', 'unetpp':'UNet++',
        'deeplab':'Compact DeepLabV3+', 'unetwide':'Wide U-Net', 'bcskg':'BCRG-Net'}
SHORT = {'unet':'U-Net', 'attention':'Skip-att.', 'attention_original':'Attention',
         'unetpp':'UNet++', 'deeplab':'DeepLab', 'unetwide':'Wide U-Net', 'bcskg':'BCRG-Net'}
# Project-specific green/purple palette, fixed method mapping across all panels.
COL = {'unet':'#9DC499', 'attention':'#A7B688', 'attention_original':'#6F9E8E',
       'unetpp':'#A987C9', 'deeplab':'#7D9A77', 'unetwide':'#765D91',
       'bcskg':'#2F6742'}
FILL = {'unet':'#E1F0E4', 'attention':'#ECF0E2', 'attention_original':'#D6EAE5',
        'unetpp':'#DCC6EA', 'deeplab':'#E1EBD7', 'unetwide':'#E3D8EF',
        'bcskg':'#C7E0D1'}
MARK = {'unet':'o','attention':'s','attention_original':'^','unetpp':'D',
        'deeplab':'v','unetwide':'P','bcskg':'X'}

def tag(method, seed):
    if method == 'unetwide':
        return f'kvasir_unet_s{seed}_r0.05_b0.1_base10'
    return f'kvasir_{method}_s{seed}_r0.05_b0.1'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def result(method, seed):
    r = read_json(RESULTS / tag(method, seed) / 'result.json')
    assert r['epochs'] == 12 and r['size'] == 64
    return r

def rows(method, seed):
    return read_json(RESULTS / tag(method, seed) / 'per_image.json')

def save(fig, name):
    from matplotlib import pyplot as plt
    fig.savefig(OUTPUT / f'{name}.pdf', facecolor='white', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{name}.png', dpi=420, facecolor='white', bbox_inches='tight')
    plt.close(fig)

def style():
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.2,
        'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,
        'ytick.labelsize':8,'legend.fontsize':7.7,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#6B7280','axes.labelcolor':'#2F4F3A',
        'text.color':'#2F4F3A','xtick.color':'#6B7280','ytick.color':'#6B7280',
        'grid.color':'#DFE7DF','grid.linewidth':0.5,
        'pdf.fonttype':42,'ps.fonttype':42})
