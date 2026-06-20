import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluation.experiment import Experiment, ExperimentConfig
from app.config import settings, PROJECT_ROOT

log_path = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'golden_run.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log_f = open(log_path, 'w', encoding='utf-8')
def log(msg):
    log_f.write(str(msg) + chr(10)); log_f.flush()

log('Start: ' + time.strftime('%Y-%m-%d %H:%M:%S'))
log('Models: chat=' + settings.chat_model + ' judge=' + settings.judge_model + ' rewrite=' + settings.rewrite_model)

exp = Experiment(test_qa_path=os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_golden.json'))
config = ExperimentConfig(name='golden_qwen35flash', top_k=5, description='Golden regression')
result = exp.run(config)

s = result.summary()
log('')
for k, v in s.items(): log(k + ': ' + str(v))

out_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'golden_qwen35flash_' + result.timestamp)
exp.save_results(result, out_dir)
with open(os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'golden_baseline.json'), 'w', encoding='utf-8') as f:
    json.dump(s, f, ensure_ascii=False, indent=2)
log('Done: ' + out_dir)
log_f.close()
