import sys, os, json, time
sys.path.insert(0, r"{}" .format(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from evaluation.experiment import Experiment, ExperimentConfig
from app.config import settings, PROJECT_ROOT

log_path = os.path.join(PROJECT_ROOT, "evaluation", "results", "compare3_run.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
lf = open(log_path, "w", encoding="utf-8")

def log(msg):
    lf.write(str(msg) + chr(10))
    lf.flush()

log("3-config comparison start: " + time.strftime("%Y-%m-%d %H:%M:%S"))
log("Chat={} Judge={} Rewrite={} Rerank={}" .format(settings.chat_model, settings.judge_model, settings.rewrite_enabled, settings.rerank_enabled))

configs = [
    ExperimentConfig("baseline", top_k=5, use_mmr=False, use_reranker=False, description="Dense only"),
    ExperimentConfig("+Rerank", top_k=5, use_mmr=False, use_reranker=True, description="Dense+Rerank"),
    ExperimentConfig("+MMR", top_k=5, use_mmr=True, use_reranker=False, description="Dense+MMR"),
]

exp = Experiment(test_qa_path=os.path.join(PROJECT_ROOT, "evaluation", "test_qa_golden.json"))

for config in configs:
    log(chr(10) + "="*60)
    log("Config: " + config.name)
    result = exp.run(config)
    s = result.summary()
    log("--- Summary ---")
    for k, v in s.items():
        log(k + ": " + str(v))
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "results", "compare_" + config.name + "_" + result.timestamp)
    exp.save_results(result, out_dir)

log(chr(10) + "ALL DONE: " + time.strftime("%H:%M:%S"))
lf.close()
