import json, copy
from pathlib import Path
out = Path(r'C:\tools\strata-disk\build-cuda')
cfg = json.loads((out / 'followup-disk-nohash.json').read_text())
cfg['args'][cfg['args'].index('--conversation-cache-disk-staging-mib') + 1] = '512'
cfg['args'][cfg['args'].index('--conversation-cache-spill-dir') + 1] = r'D:\strata-disk-cache-followup-staging512'
cfg['log'] = str(out / 'followup-staging512.log')
(out / 'followup-staging512.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
print('ok')
