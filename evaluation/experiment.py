"""
========== 对比实验框架 ==========
支持在不同配置下运行 RAG pipeline，对比评估结果。
用于回答"改了一个参数，效果变好了还是变差了？"这个问题。
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict

import pandas as pd

from app.retriever import retrieve
from app.generator import generate_answer, format_context
from evaluation.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from app.config import settings, PROJECT_ROOT


@dataclass
class ExperimentConfig:
    """
    实验配置：描述一组 RAG 参数组合。
    """
    name: str                          # 配置名称（如 "baseline", "chunk300"）
    chunk_size: int = 500              # 分块大小
    chunk_overlap: int = 100           # 分块重叠
    top_k: int = settings.top_k        # 检索返回数（默认从全局配置读取）
    use_mmr: bool = False              # 是否使用 MMR
    use_reranker: bool = False         # 是否使用 Rerank API 重排序
    use_rewrite: bool = False          # 是否使用 Query Rewrite 重写查询
    description: str = ""              # 配置描述

    def apply(self):
        """将本配置应用到全局设置中"""
        settings.chunk_size = self.chunk_size
        settings.chunk_overlap = self.chunk_overlap
        settings.top_k = self.top_k


@dataclass
class EvalResult:
    """单条测试样本的评估结果"""
    question: str
    golden_answer: str
    answer: str
    retrieved_chunks: List[str]
    faithfulness_score: float
    relevancy_score: float
    precision_score: float
    recall_score: float
    latency: float                     # 响应时间（秒）


@dataclass
class ExperimentResult:
    """一组实验的完整结果"""
    config: ExperimentConfig
    results: List[EvalResult] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(r.faithfulness_score for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_relevancy(self) -> float:
        return sum(r.relevancy_score for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_precision(self) -> float:
        return sum(r.precision_score for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_recall(self) -> float:
        return sum(r.recall_score for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_latency(self) -> float:
        return sum(r.latency for r in self.results) / len(self.results) if self.results else 0

    def summary(self) -> dict:
        """返回汇总指标"""
        return {
            "config": self.config.name,
            "faithfulness": round(self.avg_faithfulness, 4),
            "answer_relevancy": round(self.avg_relevancy, 4),
            "context_precision": round(self.avg_precision, 4),
            "context_recall": round(self.avg_recall, 4),
            "latency_avg": round(self.avg_latency, 2),
            "sample_count": len(self.results),
        }


class Experiment:
    """
    对比实验管理器。

    用法：
        exp = Experiment(test_qa_path="evaluation/test_qa.json")
        config = ExperimentConfig(name="baseline", chunk_size=500, description="默认配置")
        result = exp.run(config)
        result.summary()
    """

    def __init__(self, test_qa_path: str = None):
        """
        初始化实验管理器。

        参数：
            test_qa_path: 测试 QA 数据集的 JSON 文件路径，默认从项目根目录读取
        """
        if test_qa_path is None:
            test_qa_path = os.path.join(PROJECT_ROOT, "evaluation", "test_qa.json")
        with open(test_qa_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        在指定配置下运行完整的评估流程。

        流程：
        1. 保存当前配置
        2. 应用实验配置
        3. 对每个测试问题：检索 → 生成 → 计算 4 个指标
        4. 恢复原始配置
        5. 返回汇总结果
        """
        # 保存原始配置值，实验结束后恢复
        original = {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
        }
        try:
            config.apply()
            print(f"▶ 运行实验：{config.name} ({config.description})")
            print(f"   参数：chunk_size={config.chunk_size}, "
                  f"overlap={config.chunk_overlap}, top_k={config.top_k}, "
                  f"mmr={config.use_mmr}, reranker={config.use_reranker}, "
                  f"rewrite={config.use_rewrite}")

            results = []
            for i, item in enumerate(self.test_data):
                question = item["question"]
                golden = item["golden_answer"]

                start = time.time()

                # Step 1: 检索
                docs = retrieve(
                    query=question,
                    top_k=config.top_k,
                    use_mmr=config.use_mmr,
                    use_reranker=config.use_reranker,
                    use_rewrite=config.use_rewrite,
                )
                context_text = format_context(docs)
                retrieved_chunks = [d.page_content for d in docs]

                # Step 2: 生成
                answer = generate_answer(query=question, docs=docs)

                # Step 3: 评估
                latency = time.time() - start

                fs = faithfulness(answer, context_text)
                ar = answer_relevancy(question, answer)
                cp = context_precision(question, retrieved_chunks)
                cr = context_recall(golden, context_text)

                results.append(EvalResult(
                    question=question,
                    golden_answer=golden,
                    answer=answer,
                    retrieved_chunks=retrieved_chunks,
                    faithfulness_score=fs,
                    relevancy_score=ar,
                    precision_score=cp,
                    recall_score=cr,
                    latency=latency,
                ))

                print(f"   [{i+1}/{len(self.test_data)}] {question[:30]}... "
                      f"F={fs:.2f} R={ar:.2f} P={cp:.2f} Re={cr:.2f}")

            return ExperimentResult(config=config, results=results)
        finally:
            # 恢复原始配置，避免污染后续实验
            settings.chunk_size = original["chunk_size"]
            settings.chunk_overlap = original["chunk_overlap"]
            settings.top_k = original["top_k"]

    def compare(self, configs: List[ExperimentConfig]) -> pd.DataFrame:
        """
        对比多组配置。
        返回一个 DataFrame，每行是一组配置的汇总指标。
        """
        summaries = []
        for config in configs:
            result = self.run(config)
            summaries.append(result.summary())

        df = pd.DataFrame(summaries)
        print("\n" + "=" * 70)
        print("📊 实验对比结果")
        print("=" * 70)
        print(df.to_string(index=False))
        return df
