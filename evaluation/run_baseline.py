"""
Baseline evaluation script.
Usage:
  python -m evaluation.run_baseline                  # Full: 10 questions × 4 configs
  python -m evaluation.run_baseline --smoke          # Smoke: 3 questions × 1 config
  python -m evaluation.run_baseline --smoke --top-n 5  # Smoke: 5 questions × 1 config
"""

import sys
import os
import io
import argparse
import copy

# Fix Windows GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.experiment import Experiment, ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode: only 3 samples × 1 config")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of samples in smoke mode (default: 3)")
    args = parser.parse_args()

    exp = Experiment()

    if args.smoke:
        # 冒烟模式：少量样本 + 1 组配置，仅验证链路通断
        top_n = min(args.top_n, len(exp.test_data))
        exp.test_data = exp.test_data[:top_n]
        configs = [
            ExperimentConfig(
                name="baseline",
                chunk_size=500,
                chunk_overlap=100,
                top_k=5,
                use_mmr=False,
                use_reranker=False,
                use_rewrite=False,
                description="Smoke test (dense retrieval only)",
            ),
        ]
        label = f"SMOKE TEST: {top_n} Questions × 1 Config"
    else:
        configs = [
            ExperimentConfig(
                name="baseline",
                chunk_size=500,
                chunk_overlap=100,
                top_k=5,
                use_mmr=False,
                use_reranker=False,
                use_rewrite=False,
                description="Dense retrieval only, no optimization",
            ),
            ExperimentConfig(
                name="+Rerank",
                chunk_size=500,
                chunk_overlap=100,
                top_k=5,
                use_mmr=False,
                use_reranker=True,
                use_rewrite=False,
                description="Dense + qwen3-rerank",
            ),
            ExperimentConfig(
                name="+Rewrite",
                chunk_size=500,
                chunk_overlap=100,
                top_k=5,
                use_mmr=False,
                use_reranker=False,
                use_rewrite=True,
                description="LLM query rewrite + Dense",
            ),
            ExperimentConfig(
                name="+Rewrite+Rerank",
                chunk_size=500,
                chunk_overlap=100,
                top_k=5,
                use_mmr=False,
                use_reranker=True,
                use_rewrite=True,
                description="Rewrite + Dense + qwen3-rerank",
            ),
        ]
        label = "FULL: 10 Questions × 4 Configs"

    print("=" * 70)
    print(f"[RAG Baseline Evaluation] {label}")
    print("=" * 70)

    df = exp.compare(configs)

    print("\n" + "=" * 70)
    print("[Detailed Results]")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"\n  [{row['config']}]")
        print(f"    Faithfulness:     {row['faithfulness']:.4f}")
        print(f"    Answer Relevancy: {row['answer_relevancy']:.4f}")
        print(f"    Context Precision:{row['context_precision']:.4f}")
        print(f"    Context Recall:   {row['context_recall']:.4f}")
        print(f"    Avg Latency:      {row['latency_avg_s']:.2f}s")
        print(f"    Samples:          {row['sample_count']}")


if __name__ == "__main__":
    main()
