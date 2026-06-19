"""
========== HTML Report Generator ==========
Generates self-contained HTML reports from experiment results.

Usage:
  python -m evaluation.reporter results/benchmark_20260101_120000/
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import PROJECT_ROOT


def _load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _metric_row(label: str, value: Any, ci: str = None) -> str:
    ci_display = f' <span style="color:#888">CI95: {ci}</span>' if ci else ''
    return f'<tr><td>{label}</td><td><strong>{value}</strong>{ci_display}</td></tr>'


def _category_table(rows: list) -> str:
    """Build a breakdown table by category."""
    html = '<table><tr><th>Category</th><th>Count</th><th>Faithfulness</th><th>MRR</th><th>NDCG@5</th><th>Hit Rate</th></tr>'
    for row in rows:
        html += f'<tr><td>{row["category"]}</td><td>{row["count"]}</td><td>{row.get("faithfulness","-")}</td><td>{row.get("mrr","-")}</td><td>{row.get("ndcg","-")}</td><td>{row.get("hit_rate","-")}</td></tr>'
    html += '</table>'
    return html


def generate_report(results_dir: str, output_path: str = None):
    """
    Generate a self-contained HTML report from experiment results.

    Args:
        results_dir: Path to directory containing summary.json, raw_results.json, config.json
        output_path: Optional output HTML path (default: results_dir/report.html)
    """
    if not os.path.isdir(results_dir):
        print(f"ERROR: Directory not found: {results_dir}")
        return

    # Load files
    summary_path = os.path.join(results_dir, 'summary.json')
    raw_path = os.path.join(results_dir, 'raw_results.json')
    config_path = os.path.join(results_dir, 'config.json')
    csv_path = os.path.join(results_dir, 'results.csv')

    summary = _load_json(summary_path) if os.path.exists(summary_path) else {}
    raw = _load_json(raw_path) if os.path.exists(raw_path) else []
    config = _load_json(config_path) if os.path.exists(config_path) else {}

    # Generate metrics table
    metrics_html = ''
    metric_labels = [
        ('faithfulness', 'Faithfulness'),
        ('factual_faithfulness', 'Factual Faithfulness'),
        ('hallucination', 'Hallucination Score'),
        ('answer_relevancy', 'Answer Relevancy'),
        ('context_precision', 'Context Precision'),
        ('context_recall', 'Context Recall'),
        ('mrr', 'MRR'),
        ('ndcg@5', 'NDCG@5'),
        ('hit_rate@5', 'Hit Rate@5'),
        ('map@5', 'MAP@5'),
        ('latency_avg_s', 'Avg Latency (s)'),
        ('latency_p50_s', 'P50 Latency (s)'),
        ('latency_p95_s', 'P95 Latency (s)'),
    ]
    for key, label in metric_labels:
        if key in summary or f'avg_{key}' in summary:
            val = summary.get(key, '')
            ci = summary.get(f'{key}_ci95', '')
            if val != '':
                metrics_html += _metric_row(label, val, ci)

    # Category breakdown
    category_html = ''
    if raw:
        from collections import defaultdict
        cats = defaultdict(lambda: {'count': 0, 'ff': [], 'mrr': [], 'ndcg': [], 'hr': []})
        for item in raw:
            cat = item.get('category', 'unknown')
            cats[cat]['count'] += 1
            if 'faithfulness_score' in item:
                cats[cat]['ff'].append(item['faithfulness_score'])
            if 'mrr' in item:
                cats[cat]['mrr'].append(item['mrr'])
            if 'ndcg_at_5' in item:
                cats[cat]['ndcg'].append(item['ndcg_at_5'])
            if 'hit_rate' in item:
                cats[cat]['hr'].append(item['hit_rate'])

        cat_rows = []
        import numpy as np
        for cat, data in sorted(cats.items()):
            cat_rows.append({
                'category': cat,
                'count': data['count'],
                'faithfulness': f"{np.mean(data['ff']):.3f}" if data['ff'] else '-',
                'mrr': f"{np.mean(data['mrr']):.3f}" if data['mrr'] else '-',
                'ndcg': f"{np.mean(data['ndcg']):.3f}" if data['ndcg'] else '-',
                'hit_rate': f"{np.mean(data['hr']):.3f}" if data['hr'] else '-',
            })
        category_html = _category_table(cat_rows)

    # Build HTML
    config_display = json.dumps(config, indent=2, ensure_ascii=False) if config else 'N/A'
    sample_count = summary.get('sample_count', len(raw))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Evaluation Report - {os.path.basename(results_dir)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a3a6b; border-bottom: 3px solid #1a3a6b; padding-bottom: 10px; }}
  h2 {{ color: #2a5a9b; margin-top: 30px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f4f8; font-weight: 600; }}
  tr:hover {{ background: #f8fafc; }}
  .good {{ color: #16a34a; font-weight: bold; }}
  .warn {{ color: #ea580c; font-weight: bold; }}
  .bad {{ color: #dc2626; font-weight: bold; }}
  .meta {{ color: #888; font-size: 0.9em; }}
  pre {{ background: #f8fafc; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 0.85em; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>RAG 系统评估报告</h1>
<div class="meta">生成时间: {timestamp} | 实验ID: {os.path.basename(results_dir)} | 样本数: {sample_count}</div>

<div class="card">
<h2>配置</h2>
<pre>{config_display}</pre>
</div>

<div class="card">
<h2>汇总指标</h2>
<table>{metrics_html}</table>
</div>

<div class="card">
<h2>按领域分类</h2>
{category_html}
<div class="meta">注: 仅当原始结果包含 category 字段时显示</div>
</div>

<div class="card">
<h2>问答详情 (前20条)</h2>
'''

    # Add question details
    for i, item in enumerate(raw[:20]):
        q = item.get('question', '')[:100]
        a = item.get('answer', '')[:300]
        golden = item.get('golden_answer', '')[:200]
        fs = item.get('faithfulness_score', '-')
        mrr_val = item.get('mrr', '-')

        fs_class = 'good' if isinstance(fs, (int, float)) and fs >= 0.7 else ('warn' if isinstance(fs, (int, float)) and fs >= 0.4 else 'bad')
        html += f'''
<div style="border:1px solid #e5e7eb; border-radius:6px; padding:12px; margin:8px 0;">
  <strong>Q{i+1}:</strong> {q}<br>
  <span style="color:#888; font-size:0.9em;">Gold: {golden[:100]}...</span><br>
  <strong>A:</strong> {a[:200]}...<br>
  <span>Faith: <span class="{fs_class}">{fs}</span> | MRR: {mrr_val}</span>
</div>'''

    html += f'''
</div>

<div class="footer">
  <p>RAG Evaluation System v2.0 | Generated by evaluation/reporter.py</p>
  <p>Project root: {PROJECT_ROOT}</p>
</div>
</body>
</html>'''

    if output_path is None:
        output_path = os.path.join(results_dir, 'report.html')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report generated: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        # Try to find latest results
        results_root = os.path.join(PROJECT_ROOT, 'evaluation', 'results')
        if os.path.isdir(results_root):
            dirs = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])
            if dirs:
                latest = os.path.join(results_root, dirs[-1])
                print(f"Using latest results: {latest}")
                generate_report(latest)
                return
        print("Usage: python -m evaluation.reporter <results_directory>")
        sys.exit(1)

    generate_report(sys.argv[1])


if __name__ == '__main__':
    main()
