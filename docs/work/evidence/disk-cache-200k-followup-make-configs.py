import json
import copy
from pathlib import Path

base_disk = json.loads(Path(r'C:\tools\strata-disk\docs\work\evidence\disk-cache-200k-diskonly-config.json').read_text())
base_l1 = json.loads(Path(r'C:\tools\strata-disk\docs\work\evidence\disk-cache-200k-config.json').read_text())
out = Path(r'C:\tools\strata-disk\build-cuda')

def arg_value(cfg, flag):
    a = cfg['args']
    return a[a.index(flag) + 1]

def set_arg(cfg, flag, value):
    a = cfg['args']
    if flag in a:
        a[a.index(flag) + 1] = value
    else:
        a.extend([flag, value])

def drop_arg(cfg, flag, takes_value=True):
    a = cfg['args']
    if flag in a:
        i = a.index(flag)
        del a[i:i + (2 if takes_value else 1)]

def common(cfg, name, spill=None, log=None):
    cfg['log'] = str(out / log)
    cfg['port'] = 8099
    cfg['host'] = '127.0.0.1'
    cfg['env'].pop('STRATA_STATE_HASH', None)
    if spill:
        set_arg(cfg, '--conversation-cache-spill-dir', spill)
    (out / name).write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print(name, 'ok')

# A baseline: no RAM cache, no disk tier
b = copy.deepcopy(base_disk)
drop_arg(b, '--conversation-cache-disk-only', takes_value=False)
for f in ('--conversation-cache-disk-mib', '--conversation-cache-spill-dir',
          '--conversation-cache-disk-min-free-mib', '--conversation-cache-disk-staging-mib'):
    drop_arg(b, f)
set_arg(b, '--conversation-cache-mib', '0')
common(b, 'followup-baseline.json', log='followup-baseline.log')

# A disk arm + C: disk-only, no STATE_HASH, fresh spill dir
d = copy.deepcopy(base_disk)
common(d, 'followup-disk-nohash.json', spill=r'D:\strata-disk-cache-followup-disk',
       log='followup-disk-nohash.log')

# B1: L1+L2, L1 admission floor forces the disk (physical-RAM path)
b1 = copy.deepcopy(base_l1)
set_arg(b1, '--conversation-cache-min-free-mib', '16000')
common(b1, 'followup-b1-ram-refusal.json', spill=r'D:\strata-disk-cache-followup-b1',
       log='followup-b1.log')

# B2: L1+L2, L1 budget smaller than the 194K snapshot (budget path)
b2 = copy.deepcopy(base_l1)
set_arg(b2, '--conversation-cache-mib', '2048')
common(b2, 'followup-b2-budget-refusal.json', spill=r'D:\strata-disk-cache-followup-b2',
       log='followup-b2.log')
