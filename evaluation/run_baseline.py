"""
Baseline evaluation script.
Usage: ../.venv/Scripts/python.exe -m evaluation.run_baseline
"""

import sys
import os
import io

# Fix Windows GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.experiment import Experiment, ExperimentConfig


def main():
    exp = Experiment()

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
            description="Dense + gte-rerank",
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
            description="Rewrite + Dense + gte-rerank",
        ),
    ]

    print("=" * 70)
    print("[RAG Baseline Evaluation] 10 Questions x 4 Configs")
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
        print(f"    Avg Latency:      {row['latency_avg']:.2f}s")
        print(f"    Samples:          {row['sample_count']}")


if __name__ == "__main__":
    main()
