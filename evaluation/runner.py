"""
========== Unified CLI Runner ==========
One entry point for all evaluation modes.

Usage:
  python -m evaluation.runner benchmark               # 60 questions x baseline
  python -m evaluation.runner benchmark --full         # 60 questions x 3 configs
  python -m evaluation.runner benchmark --quick        # 15 questions x 1 config
  python -m evaluation.runner benchmark --dataset golden  # Golden regression set
  python -m evaluation.runner golden                   # Alias: golden regression
  python -m evaluation.runner robustness               # Robustness test
  python -m evaluation.runner oos                      # Out-of-scope test
  python -m evaluation.runner report <results_dir>     # Generate HTML report
"""

import sys
import os
import io
import json
import argparse
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.experiment import Experiment, ExperimentConfig, paired_ttest
from app.config import settings, PROJECT_ROOT


def _default_configs() -> list:
    """Return the 3 default experiment configurations (Rewrite disabled server-side)."""
    return [
        ExperimentConfig(name="baseline", top_k=5, use_mmr=False, use_reranker=False, use_rewrite=False,
                         description="Dense retrieval only"),
        ExperimentConfig(name="+Rerank", top_k=5, use_mmr=False, use_reranker=True, use_rewrite=False,
                         description="Dense + qwen3-rerank"),
        ExperimentConfig(name="+MMR", top_k=5, use_mmr=True, use_reranker=False, use_rewrite=False,
                         description="Dense + MMR (diversity)"),
    ]


def cmd_benchmark(args):
    """Run Layer 1 benchmark evaluation."""
    dataset = args.dataset
    qa_paths = {
        'full': os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_v2.json'),
        'golden': os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_golden.json'),
    }
    qa_path = qa_paths.get(dataset, qa_paths['full'])

    exp = Experiment(test_qa_path=qa_path)

    if args.quick:
        configs = [ExperimentConfig(name="baseline", top_k=5, description="Quick smoke test")]
        exp.test_data = exp.test_data[:min(args.top_n, len(exp.test_data))]
        label = f"QUICK: {len(exp.test_data)} Q x 1 Config"
    elif args.full:
        configs = _default_configs()
        label = f"FULL: {len(exp.test_data)} Q x {len(configs)} Configs"
    else:
        configs = [ExperimentConfig(name="baseline", top_k=5, description="Dense retrieval only")]
        label = f"BENCHMARK: {len(exp.test_data)} Q x 1 Config"

    print("=" * 70)
    print(f"[RAG Evaluation] {label}")
    print("=" * 70)

    df = exp.compare(configs)

    # Print detailed results
    print("\n" + "=" * 70)
    print("[Summary]")
    print("=" * 70)
    for _, row in df.iterrows():
        print(f"\n  [{row['config']}]")
        print(f"    Faithfulness:       {row.get('faithfulness', 'N/A')}")
        print(f"    Factual Faith:      {row.get('factual_faithfulness', 'N/A')}")
        print(f"    Answer Relevancy:   {row.get('answer_relevancy', 'N/A')}")
        print(f"    Context Precision:  {row.get('context_precision', 'N/A')}")
        print(f"    Context Recall:     {row.get('context_recall', 'N/A')}")
        print(f"    MRR:                {row.get('mrr', 'N/A')}")
        print(f"    NDCG@5:             {row.get('ndcg@5', 'N/A')}")
        print(f"    Hit Rate@5:         {row.get('hit_rate@5', 'N/A')}")
        print(f"    Avg Latency:        {row.get('latency_avg_s', 'N/A')}s")
        print(f"    Samples:            {row.get('sample_count', 'N/A')}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'benchmark_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {out_dir}")


def cmd_golden(args):
    """Run golden regression test."""
    qa_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_golden.json')
    if not os.path.exists(qa_path):
        print(f"ERROR: Golden QA file not found: {qa_path}")
        print("Run: python evaluation/_build_data.py first")
        sys.exit(1)

    exp = Experiment(test_qa_path=qa_path)
    config = ExperimentConfig(name="golden_baseline", top_k=5, description="Golden regression")
    result = exp.run(config)

    print("\n" + "=" * 70)
    print("[Golden Regression Results]")
    print("=" * 70)
    s = result.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Check for regressions (compare with saved baseline if exists)
    baseline_path = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'golden_baseline.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        print("\n[Regression Check vs saved baseline]")
        for metric in ['faithfulness', 'answer_relevancy', 'mrr']:
            if metric in baseline and metric in s:
                old = baseline[metric]
                new = s[metric]
                delta = new - old
                flag = "WARNING" if delta < -0.05 else "OK"
                print(f"  {metric}: {old:.4f} -> {new:.4f} ({delta:+.4f}) [{flag}]")

    # Save as new baseline
    exp.save_results(result, os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'golden_{result.timestamp}'))
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"Updated golden baseline saved.")


def cmd_robustness(args):
    """Run robustness test (typo + spoken variants)."""
    perturbed_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_perturbed.json')
    if not os.path.exists(perturbed_path):
        print(f"ERROR: Perturbed QA file not found: {perturbed_path}")
        sys.exit(1)

    # First, get baseline on original questions
    orig_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_v2.json')
    exp_orig = Experiment(test_qa_path=orig_path)
    # Filter to only questions that have perturbations
    perturb_ids = set()
    with open(perturbed_path, 'r', encoding='utf-8') as f:
        perturbed_data = json.load(f)
    for item in perturbed_data:
        base_id = item['question_id'].replace('_typo', '').replace('_spoken', '')
        perturb_ids.add(base_id)

    exp_orig.test_data = [item for item in exp_orig.test_data if item['question_id'] in perturb_ids]
    config = ExperimentConfig(name="original", top_k=5, description="Original questions")
    print("=" * 70)
    print("[Phase 1] Baseline on original questions")
    print("=" * 70)
    result_orig = exp_orig.run(config)

    # Then, run on perturbed
    exp_pert = Experiment(test_qa_path=perturbed_path)
    config2 = ExperimentConfig(name="perturbed", top_k=5, description="Perturbed questions")
    print("\n" + "=" * 70)
    print("[Phase 2] Robustness test on perturbed questions")
    print("=" * 70)
    result_pert = exp_pert.run(config2)

    # Compare
    print("\n" + "=" * 70)
    print("[Robustness Comparison]")
    print("=" * 70)
    metrics = [
        ('faithfulness_score', 'Faithfulness'),
        ('factual_faithfulness_score', 'Factual Faith'),
        ('relevancy_score', 'Relevancy'),
        ('mrr', 'MRR'),
        ('ndcg_at_5', 'NDCG@5'),
        ('hit_rate', 'Hit Rate'),
    ]
    print(f"{'Metric':<20} {'Original':>10} {'Perturbed':>10} {'Degradation':>12}")
    print("-" * 55)
    for attr, label in metrics:
        orig_val = getattr(result_orig, f'avg_{attr}') if hasattr(result_orig, f'avg_{attr}') else result_orig._mean(attr)
        pert_val = getattr(result_pert, f'avg_{attr}') if hasattr(result_pert, f'avg_{attr}') else result_pert._mean(attr)
        degradation = orig_val - pert_val
        print(f"{label:<20} {orig_val:>10.4f} {pert_val:>10.4f} {degradation:>+11.4f}")

    # Summary degradation
    print(f"\nAvg Faithfulness Degradation: {result_orig.avg_faithfulness - result_pert.avg_faithfulness:+.4f}")
    print(f"Avg MRR Degradation: {result_orig.avg_mrr - result_pert.avg_mrr:+.4f}")


def cmd_oos(args):
    """Run out-of-scope test - verify system properly refuses unknown questions."""
    oos_path = os.path.join(PROJECT_ROOT, 'evaluation', 'test_qa_oos.json')
    if not os.path.exists(oos_path):
        print(f"ERROR: OOS QA file not found: {oos_path}")
        sys.exit(1)

    with open(oos_path, 'r', encoding='utf-8') as f:
        oos_data = json.load(f)

    print("=" * 70)
    print(f"[Out-of-Scope Test] {len(oos_data)} questions")
    print("=" * 70)

    from app.retriever import retrieve
    from app.generator import generate_answer, format_context

    config = ExperimentConfig(name="oos_test", top_k=5)
    refused = 0
    hallucinated = 0

    for i, item in enumerate(oos_data):
        question = item['question']
        print(f"\n[{i+1}/{len(oos_data)}] {question}")

        docs = retrieve(query=question, top_k=config.top_k, use_mmr=False, use_reranker=False, use_rewrite=False)
        answer = generate_answer(query=question, docs=docs)
        context = format_context(docs)

        # Check for refusal signals
        refusal_signals = ['没有找到', '无法回答', '不在', '超出', '未涵盖', '建议', '请访问', '请参考', '抱歉']
        is_refusal = any(signal in answer for signal in refusal_signals)

        if is_refusal:
            refused += 1
            print(f"  Result: REFUSED (correct)")
        else:
            print(f"  Result: ANSWERED (potential hallucination)")
            print(f"  Answer: {answer[:200]}...")
            hallucinated += 1

    print(f"\n{'='*50}")
    print(f"OOS Results:")
    print(f"  Correctly Refused: {refused}/{len(oos_data)} ({refused/len(oos_data)*100:.0f}%)")
    print(f"  Hallucinated:      {hallucinated}/{len(oos_data)} ({hallucinated/len(oos_data)*100:.0f}%)")


def cmd_report(args):
    """Generate HTML report from results directory."""
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    try:
        from evaluation.reporter import generate_report
        generate_report(results_dir)
    except ImportError:
        print("Reporter module not available yet.")
        print(f"Results directory: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='RAG Evaluation Runner')
    sub = parser.add_subparsers(dest='command', help='Evaluation mode')

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Run benchmark evaluation')
    p_bench.add_argument('--full', action='store_true', help='Run all 4 configs')
    p_bench.add_argument('--quick', action='store_true', help='Quick smoke test')
    p_bench.add_argument('--dataset', default='full', choices=['full', 'golden'], help='Dataset to use')
    p_bench.add_argument('--top-n', type=int, default=3, help='Questions in quick mode')

    # golden
    sub.add_parser('golden', help='Run golden regression test')

    # robustness
    sub.add_parser('robustness', help='Run robustness test')

    # oos
    sub.add_parser('oos', help='Run out-of-scope test')

    # report
    p_report = sub.add_parser('report', help='Generate HTML report')
    p_report.add_argument('results_dir', help='Path to results directory')

    args = parser.parse_args()

    if args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'golden':
        cmd_golden(args)
    elif args.command == 'robustness':
        cmd_robustness(args)
    elif args.command == 'oos':
        cmd_oos(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
