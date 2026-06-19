"""
========== 对比实验框架 ==========
支持在不同配置下运行 RAG pipeline，对比评估结果。
用于回答"改了一个参数，效果变好了还是变差了？"这个问题。
"""

import json
import os
import time
import math
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import pandas as pd
import numpy as np

from app.retriever import retrieve
from app.generator import generate_answer, format_context
from evaluation.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from evaluation.metrics_retrieval import compute_all_retrieval_metrics
from evaluation.metrics_factual import factual_faithfulness, hallucination_check
from app.config import settings, PROJECT_ROOT


@dataclass
class ExperimentConfig:
    """
    实验配置：描述一组 RAG 参数组合。

    注意：chunk_size / chunk_overlap 是索引时参数，变更后需重建向量库才能生效。
    对比实验应先用不同 chunk_size 分别运行 init_db.py 建立独立数据库，
    或仅对比运行时可切换的参数（top_k / use_mmr / use_reranker / use_rewrite）。
    """
    name: str                          # 配置名称（如 "baseline", "chunk300"）
    chunk_size: int = 500              # 分块大小（仅供记录，不影响运行时检索）
    chunk_overlap: int = 100           # 分块重叠（仅供记录，不影响运行时检索）
    top_k: int = settings.top_k        # 检索返回数（运行时生效，传给 retrieve）
    use_mmr: bool = False              # 是否使用 MMR
    use_reranker: bool = False         # 是否使用 Rerank API 重排序
    use_rewrite: bool = False          # 是否使用 Query Rewrite 重写查询
    description: str = ""              # 配置描述


@dataclass
class EvalResult:
    """Single question evaluation result with 8 metric dimensions."""
    question_id: str = ""
    question: str = ""
    golden_answer: str = ""
    answer: str = ""
    retrieved_chunks: List[str] = field(default_factory=list)
    faithfulness_score: float = 0.0
    factual_faithfulness_score: float = 0.0
    hallucination_score: float = 0.0
    relevancy_score: float = 0.0
    precision_score: float = 0.0
    recall_score: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate: float = 0.0
    map_at_5: float = 0.0
    latency: float = 0.0
    category: str = ""
    difficulty: str = ""
    question_type: str = ""


@dataclass
class ExperimentResult:
    """Complete experiment result with statistical analysis."""
    config: ExperimentConfig
    results: List[EvalResult] = field(default_factory=list)
    run_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.run_id:
            self.run_id = self.timestamp

    def _mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    def _bootstrap_ci(self, attr: str, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        vals = np.array([getattr(r, attr) for r in self.results])
        if len(vals) < 3:
            return (self._mean(attr), self._mean(attr))
        rng = np.random.RandomState(42)
        means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n_bootstrap)]
        lower = float(np.percentile(means, alpha / 2 * 100))
        upper = float(np.percentile(means, (1 - alpha / 2) * 100))
        return (round(lower, 4), round(upper, 4))

    @property
    def avg_faithfulness(self): return self._mean('faithfulness_score')
    @property
    def avg_factual_faithfulness(self): return self._mean('factual_faithfulness_score')
    @property
    def avg_hallucination(self): return self._mean('hallucination_score')
    @property
    def avg_relevancy(self): return self._mean('relevancy_score')
    @property
    def avg_precision(self): return self._mean('precision_score')
    @property
    def avg_recall(self): return self._mean('recall_score')
    @property
    def avg_mrr(self): return self._mean('mrr')
    @property
    def avg_ndcg(self): return self._mean('ndcg_at_5')
    @property
    def avg_hit_rate(self): return self._mean('hit_rate')
    @property
    def avg_map(self): return self._mean('map_at_5')
    @property
    def avg_latency(self): return self._mean('latency')

    def summary(self) -> dict:
        ci_f = self._bootstrap_ci('faithfulness_score')
        return {
            "config": self.config.name,
            "sample_count": len(self.results),
            "faithfulness": round(self.avg_faithfulness, 4),
            "faithfulness_ci95": f"[{ci_f[0]}, {ci_f[1]}]",
            "factual_faithfulness": round(self.avg_factual_faithfulness, 4),
            "hallucination": round(self.avg_hallucination, 4),
            "answer_relevancy": round(self.avg_relevancy, 4),
            "context_precision": round(self.avg_precision, 4),
            "context_recall": round(self.avg_recall, 4),
            "mrr": round(self.avg_mrr, 4),
            "ndcg@5": round(self.avg_ndcg, 4),
            "hit_rate@5": round(self.avg_hit_rate, 4),
            "map@5": round(self.avg_map, 4),
            "latency_avg_s": round(self.avg_latency, 2),
            "latency_p50_s": round(float(np.median([r.latency for r in self.results])), 2) if self.results else 0,
            "latency_p95_s": round(float(np.percentile([r.latency for r in self.results], 95)), 2) if self.results else 0,
        }


# ====== Significance Testing ======

def paired_ttest(result_a: ExperimentResult, result_b: ExperimentResult, metric: str) -> dict:
    """Paired t-test between two experiment results on a specific metric."""
    vals_a = np.array([getattr(r, metric) for r in result_a.results])
    vals_b = np.array([getattr(r, metric) for r in result_b.results])
    if len(vals_a) != len(vals_b):
        return {'error': 'Sample sizes differ'}
    diff = vals_a - vals_b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff == 0:
        p_value = 0.0 if mean_diff != 0 else 1.0
        t_stat = 0.0
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        from math import erf
        def norm_cdf(x): return 0.5 * (1 + erf(x / 1.4142135623730951))
        p_value = 2 * (1 - norm_cdf(abs(t_stat)))
    return {
        'metric': metric, 'mean_diff': round(mean_diff, 4),
        't_statistic': round(t_stat, 4), 'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'ci95_diff': [round(mean_diff - 1.96 * std_diff / math.sqrt(n), 4),
                      round(mean_diff + 1.96 * std_diff / math.sqrt(n), 4)],
    }


# ====== Experiment Runner ======

class Experiment:
    """
    Enhanced experiment manager supporting 8-metric evaluation and statistical analysis.

    Usage:
        exp = Experiment(test_qa_path="evaluation/test_qa_v2.json")
        config = ExperimentConfig(name="baseline", top_k=5)
        result = exp.run(config)
        print(result.summary())
    """

    def __init__(self, test_qa_path: str = None):
        if test_qa_path is None:
            test_qa_path = os.path.join(PROJECT_ROOT, "evaluation", "test_qa_v2.json")
        if not os.path.exists(test_qa_path):
            fallback = os.path.join(PROJECT_ROOT, "evaluation", "test_qa.json")
            if os.path.exists(fallback):
                test_qa_path = fallback
                print(f"[WARN] test_qa_v2.json not found, falling back to {fallback}")
            else:
                raise FileNotFoundError(f"No test QA file found at {test_qa_path}")
        with open(test_qa_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)
        print(f"Loaded {len(self.test_data)} test questions from {os.path.basename(test_qa_path)}")

    def run(self, config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
        """Run complete evaluation pipeline: retrieve -> generate -> evaluate (8 metrics)."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: {config.name} ({config.description})")
            print(f"  top_k={config.top_k}, mmr={config.use_mmr}, "
                  f"reranker={config.use_reranker}, rewrite={config.use_rewrite}")
            print(f"{'='*60}")

        results = []
        for i, item in enumerate(self.test_data):
            question = item.get("question", "")
            golden = item.get("golden_answer", "")
            qid = item.get("question_id", f"Q{i+1:03d}")
            key_facts = item.get("key_facts", [])
            golden_terms = item.get("golden_context_must", [])

            start = time.time()

            # Step 1: Retrieve
            docs = retrieve(
                query=question, top_k=config.top_k,
                use_mmr=config.use_mmr, use_reranker=config.use_reranker,
                use_rewrite=config.use_rewrite,
            )
            context_text = format_context(docs)
            retrieved_chunks = [d.page_content for d in docs]

            # Step 2: Generate
            answer = generate_answer(query=question, docs=docs)
            latency = time.time() - start

            # Step 3: Evaluate generation quality
            fs = faithfulness(answer, context_text)
            ff = factual_faithfulness(answer, context_text, key_facts) if key_facts else fs
            hs, _ = hallucination_check(answer, context_text)
            ar = answer_relevancy(question, answer)

            # Step 4: Evaluate retrieval quality
            cp = context_precision(question, retrieved_chunks)
            cr = context_recall(golden, context_text)
            ret_metrics = compute_all_retrieval_metrics(retrieved_chunks, golden_terms, k=5)

            result = EvalResult(
                question_id=qid, question=question, golden_answer=golden,
                answer=answer, retrieved_chunks=retrieved_chunks,
                faithfulness_score=fs, factual_faithfulness_score=ff,
                hallucination_score=hs, relevancy_score=ar,
                precision_score=cp, recall_score=cr,
                mrr=ret_metrics['mrr'], ndcg_at_5=ret_metrics['ndcg_at_5'],
                hit_rate=ret_metrics['hit_rate_at_5'], map_at_5=ret_metrics['map_at_5'],
                latency=latency,
                category=item.get('category', ''), difficulty=item.get('difficulty', ''),
                question_type=item.get('question_type', ''),
            )
            results.append(result)

            if verbose:
                print(f"  [{i+1}/{len(self.test_data)}] {qid} | F={fs:.2f} FF={ff:.2f} "
                      f"H={hs:.2f} AR={ar:.2f} | MRR={ret_metrics['mrr']:.2f} "
                      f"NDCG={ret_metrics['ndcg_at_5']:.2f} HR={ret_metrics['hit_rate_at_5']:.0f} "
                      f"| {latency:.1f}s")

        return ExperimentResult(config=config, results=results)

    def run_with_repeats(self, config: ExperimentConfig, repeats: int = 3, verbose: bool = True) -> List[ExperimentResult]:
        """Run same config multiple times for variance assessment."""
        all_runs = []
        for r in range(repeats):
            if verbose: print(f"\n--- Repeat {r+1}/{repeats} ---")
            result = self.run(config, verbose=verbose)
            result.run_id = f"{config.name}_rep{r+1}"
            all_runs.append(result)
        return all_runs

    def compare(self, configs: List[ExperimentConfig], verbose: bool = True) -> pd.DataFrame:
        """Compare multiple configurations side-by-side."""
        summaries = []
        for config in configs:
            result = self.run(config, verbose=verbose)
            summaries.append(result.summary())
        df = pd.DataFrame(summaries)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Comparison Results ({len(configs)} configs)")
            print(f"{'='*70}")
            cols = ['config','faithfulness','factual_faithfulness','answer_relevancy',
                    'context_precision','context_recall','mrr','ndcg@5','hit_rate@5',
                    'latency_avg_s','sample_count']
            available = [c for c in cols if c in df.columns]
            print(df[available].to_string(index=False))
        return df

    def save_results(self, result: ExperimentResult, output_dir: str = None):
        """Save experiment results to disk (config.json, raw_results.json, summary.json, results.csv)."""
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'results', result.run_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(result.config), f, ensure_ascii=False, indent=2)
        raw = []
        for r in result.results:
            d = asdict(r)
            d['retrieved_chunks'] = d['retrieved_chunks'][:3]
            d['answer'] = d['answer'][:500]
            raw.append(d)
        with open(os.path.join(output_dir, 'raw_results.json'), 'w', encoding='utf-8') as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(result.summary(), f, ensure_ascii=False, indent=2)
        df = pd.DataFrame([asdict(r) for r in result.results])
        df.to_csv(os.path.join(output_dir, 'results.csv'), index=False, encoding='utf-8-sig')
        print(f'Results saved to {output_dir}')
        return output_dir
