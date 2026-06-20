import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Force unbuffered
import builtins
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)
builtins.print = _flush_print

import json, time
from evaluation.experiment import Experiment, ExperimentConfig
from app.config import settings, PROJECT_ROOT

_f = open(os.path.join(PROJECT_ROOT, 'evaluation', 'results', '_golden_log.txt'), 'w', encoding='utf-8')
def log(msg):
    _f.write(msg + '\n')
    _f.flush()
    print(msg)

log(f'Golden test start: {time.strftime("%H:%M:%S")}')
log(f'Chat: {settings.chat_model}, Judge: {settings.judge_model}')

qa_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_golden.json')
exp = Experiment(test_qa_path=qa_path)
config = ExperimentConfig(name="golden_qwen35flash", top_k=5, description="Golden regression")
result = exp.run(config, verbose=True)

log('\n' + '='*70)
log('[Golden Results]')
for k, v in result.summary().items():
    log(f'  {k}: {v}')

out_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'golden_qwen35flash_{result.timestamp}')
exp.save_results(result, out_dir)
log(f'Saved to {out_dir}')
_f.close()
