import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json, time, argparse
from evaluation.experiment import Experiment, ExperimentConfig
from app.config import settings, PROJECT_ROOT

print("Starting golden regression...")
print(f"Chat model: {settings.chat_model}")
print(f"Judge model: {settings.judge_model}")

qa_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_golden.json')
exp = Experiment(test_qa_path=qa_path)
config = ExperimentConfig(name="golden_baseline", top_k=5, description="Golden regression")
result = exp.run(config)

print("\n" + "=" * 70)
print("[Results]")
print("=" * 70)
s = result.summary()
for k, v in s.items():
    print(f"  {k}: {v}")

exp.save_results(result, os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'golden_{result.timestamp}'))
baseline_path = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'golden_baseline.json')
with open(baseline_path, 'w', encoding='utf-8') as f:
    json.dump(s, f, ensure_ascii=False, indent=2)
print("Saved.")
