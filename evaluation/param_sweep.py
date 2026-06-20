"""
========== Parameter Sweep ==========
Systematic exploration of RAG hyperparameters to find optimal configuration.

Supports:
- Runtime parameters (top_k, use_mmr, use_reranker, use_rewrite)
- Index-time parameters (chunk_size, chunk_overlap) with auto-rebuild

Usage:
  python -m evaluation.param_sweep --runtime    # Sweep runtime params only
  python -m evaluation.param_sweep --full        # Full sweep including chunk rebuild
  python -m evaluation.param_sweep --repeats 3   # 3 repeats per config
"""

import sys
import os
import io
import json
import argparse
import subprocess
import copy
from datetime import datetime
from typing import List, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from evaluation.experiment import Experiment, ExperimentConfig, paired_ttest
from app.config import settings, PROJECT_ROOT


# ====== Search Space ======

RUNTIME_GRID = {
    "top_k": [3, 5, 8, 10],
    "use_reranker": [False, True],
    "use_mmr": [False, True],
}

# Smart experiment paths (avoid full combinatorial explosion)
# Note: rewrite removed — server-side REWRITE_ENABLED=false
EXPERIMENT_PATHS = [
    # Path 1: Isolate top_k
    {"top_k": 3, "use_reranker": False, "use_mmr": False, "label": "top3_baseline"},
    {"top_k": 5, "use_reranker": False, "use_mmr": False, "label": "top5_baseline"},
    {"top_k": 8, "use_reranker": False, "use_mmr": False, "label": "top8_baseline"},
    {"top_k": 10, "use_reranker": False, "use_mmr": False, "label": "top10_baseline"},
    # Path 2: Rerank effect
    {"top_k": 5, "use_reranker": True, "use_mmr": False, "label": "top5_rerank"},
    {"top_k": 8, "use_reranker": True, "use_mmr": False, "label": "top8_rerank"},
    {"top_k": 10, "use_reranker": True, "use_mmr": False, "label": "top10_rerank"},
    # Path 3: MMR effect
    {"top_k": 5, "use_reranker": False, "use_mmr": True, "label": "top5_mmr"},
    {"top_k": 8, "use_reranker": False, "use_mmr": True, "label": "top8_mmr"},
    # Path 4: Rerank + MMR
    {"top_k": 5, "use_reranker": True, "use_mmr": True, "label": "top5_rerank_mmr"},
    {"top_k": 8, "use_reranker": True, "use_mmr": True, "label": "top8_rerank_mmr"},
]

CHUNK_PATHS = [
    {"chunk_size": 300, "chunk_overlap": 50},
    {"chunk_size": 500, "chunk_overlap": 100},
    {"chunk_size": 800, "chunk_overlap": 150},
    {"chunk_size": 1000, "chunk_overlap": 200},
]


def _make_config(label: str, params: dict) -> ExperimentConfig:
    """Create ExperimentConfig from parameter dict."""
    return ExperimentConfig(
        name=label,
        top_k=params.get("top_k", 5),
        use_mmr=params.get("use_mmr", False),
        use_reranker=params.get("use_reranker", False),
        use_rewrite=False,
        chunk_size=params.get("chunk_size", 500),
        chunk_overlap=params.get("chunk_overlap", 100),
        description=f"top_k={params.get('top_k',5)} rerank={params.get('use_reranker',False)} mmr={params.get('use_mmr',False)}",
    )


def _rebuild_db(chunk_size: int, chunk_overlap: int):
    """Rebuild vector DB with new chunk parameters."""
    print(f"\n[Rebuild] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(chunk_size)
    env["CHUNK_OVERLAP"] = str(chunk_overlap)
    script = os.path.join(PROJECT_ROOT, "scripts", "init_db.py")
    if os.path.exists(script):
        result = subprocess.run(
            [sys.executable, script, "--full"],
            cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] DB rebuild failed: {result.stderr[:500]}")
            return False
        print(f"[OK] DB rebuilt with chunk_size={chunk_size}")
        return True
    else:
        print(f"[ERROR] init_db.py not found at {script}")
        return False


def sweep_runtime(args) -> pd.DataFrame:
    """Sweep runtime parameters (top_k, reranker, rewrite)."""
    exp = Experiment()
    paths = EXPERIMENT_PATHS
    if args.quick:
        paths = paths[:4]  # Just top_k sweep

    configs = [_make_config(p["label"], p) for p in paths]
    label = f"Runtime Sweep: {len(configs)} configs x {len(exp.test_data)} questions"
    print("=" * 70)
    print(f"[Parameter Sweep] {label}")
    print("=" * 70)

    all_summaries = []
    for config in configs:
        repeats = max(1, args.repeats)
        if repeats > 1:
            runs = exp.run_with_repeats(config, repeats=repeats, verbose=args.verbose)
            for run in runs:
                run.config.name = config.name  # Keep same name for grouping
            avg_summary = runs[0].summary()
            for key in avg_summary:
                if key not in ('config', 'sample_count', 'faithfulness_ci95'):
                    vals = [r.summary().get(key, 0) for r in runs]
                    if isinstance(vals[0], (int, float)):
                        avg_summary[key] = round(float(np.mean(vals)), 4)
            all_summaries.append(avg_summary)
        else:
            result = exp.run(config, verbose=args.verbose)
            all_summaries.append(result.summary())

    df = pd.DataFrame(all_summaries)
    print("\n" + "=" * 70)
    print("[Sweep Results]")
    print("=" * 70)
    display_cols = ['config', 'faithfulness', 'factual_faithfulness', 'mrr', 'ndcg@5',
                    'hit_rate@5', 'latency_avg_s', 'sample_count']
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    # Find best config
    if 'faithfulness' in df.columns:
        best_idx = df['faithfulness'].idxmax()
        print(f"\nBest config by Faithfulness: {df.iloc[best_idx]['config']} "
              f"({df.iloc[best_idx]['faithfulness']:.4f})")
    if 'mrr' in df.columns:
        best_mrr = df['mrr'].idxmax()
        print(f"Best config by MRR: {df.iloc[best_mrr]['config']} "
              f"({df.iloc[best_mrr]['mrr']:.4f})")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'sweep_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'sweep_results.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'experiment_paths': paths, 'repeats': args.repeats}, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return df


def sweep_full(args) -> pd.DataFrame:
    """Full sweep including chunk_size rebuild."""
    all_dfs = []
    for chunk_cfg in CHUNK_PATHS:
        cs = chunk_cfg["chunk_size"]
        co = chunk_cfg["chunk_overlap"]
        label_prefix = f"chunk{cs}_o{co}"
        print(f"\n{'#'*60}")
        print(f"# Testing chunk_size={cs}, chunk_overlap={co}")
        print(f"{'#'*60}")

        # Rebuild DB
        if not _rebuild_db(cs, co):
            print(f"[SKIP] chunk_size={cs} - rebuild failed")
            continue

        exp = Experiment()
        paths = EXPERIMENT_PATHS[:4] if args.quick else EXPERIMENT_PATHS[:6]
        for p in paths:
            p = copy.deepcopy(p)
            p["chunk_size"] = cs
            p["chunk_overlap"] = co
            p["label"] = f"{label_prefix}_{p['label']}"

        configs = [_make_config(p["label"], p) for p in paths]
        for config in configs:
            result = exp.run(config, verbose=args.verbose)
            s = result.summary()
            s['chunk_size'] = cs
            s['chunk_overlap'] = co
            all_dfs.append(s)

    df = pd.DataFrame(all_dfs)
    print("\n" + "=" * 70)
    print("[Full Sweep Results]")
    print("=" * 70)
    display_cols = ['config', 'chunk_size', 'faithfulness', 'mrr', 'ndcg@5', 'hit_rate@5']
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', f'fullsweep_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'full_sweep.csv'), index=False, encoding='utf-8-sig')
    print(f"\nSaved to {out_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(description='RAG Parameter Sweep')
    parser.add_argument('--runtime', action='store_true', default=True,
                        help='Sweep runtime params only (default)')
    parser.add_argument('--full', action='store_true',
                        help='Full sweep including chunk_size rebuild')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced parameter set')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeats per config (default: 1)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    args = parser.parse_args()

    if args.full:
        sweep_full(args)
    else:
        sweep_runtime(args)


if __name__ == '__main__':
    main()
