# -*- coding: utf-8 -*-
"""
CI 专用验证脚本。

两种模式：
  --mode pr   : PR 分支快速检查（无需 API Key），step 7 替换为纯 config 值校验
  --mode full : 等同 python scripts/verify.py（需 API Key + ChromaDB 已初始化）
"""
import sys, os, traceback, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = 0
fail = 0

def check(title, step_fn):
    global ok, fail
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    try:
        step_fn()
        ok += 1
        print(f"  >>> PASS")
    except Exception as e:
        fail += 1
        print(f"  >>> FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()


def run_pr_mode():
    """PR 模式：steps 1-6, 8 完全同 verify.py；step 7 替换为纯 config 值校验"""

    # ── 1. 配置加载 ──
    def step1():
        from app.config import settings, PROJECT_ROOT
        print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"  chat_model: {settings.chat_model}")
        print(f"  top_k: {settings.top_k}")
        print(f"  rerank_enabled: {settings.rerank_enabled}")
        print(f"  rewrite_enabled: {settings.rewrite_enabled}")
    check("1. 配置加载", step1)

    # ── 2. 异常体系 ──
    def step2():
        from app.exceptions import RetrievalException
        e = RetrievalException(detail="测试")
        assert e.error_code == "RETRIEVAL_ERROR"
        assert e.status_code == 503
        print(f"  error_code={e.error_code}, status_code={e.status_code}, message={e.message}")
    check("2. 异常体系", step2)

    # ── 3. API Schemas ──
    def step3():
        from app.schemas import ChatRequest
        r = ChatRequest(query="测试")
        print(f"  query={r.query}, top_k={r.top_k}, force_web_search={r.force_web_search}")
        try:
            ChatRequest()
            raise AssertionError("BUG: 缺少必填字段未报错!")
        except Exception:
            pass
        print("  参数校验: ChatRequest() 缺 query 正确抛出异常")
    check("3. API Schemas", step3)

    # ── 4. 文档加载和分块 ──
    def step4():
        from app.config import PROJECT_ROOT
        from app.document_loader import load_and_split
        doc_path = os.path.join(PROJECT_ROOT, "data", "source_docs")
        txt_files = sorted([f for f in os.listdir(doc_path) if f.endswith(".txt")])
        print(f"  找到 {len(txt_files)} 个 txt 文件")
        if not txt_files:
            print("  (无可测试的 txt 文件，跳过)")
            return
        sample = os.path.join(doc_path, txt_files[0])
        chunks = load_and_split(sample)
        print(f"  文件: {os.path.basename(sample)}")
        print(f"  chunks: {len(chunks)}")
        print(f"  chunk0 前80字符: {chunks[0].page_content[:80]}...")
        print(f"  chunk0 metadata: filename={chunks[0].metadata.get('filename')}")
    check("4. 文档加载和分块", step4)

    # ── 5. Prompt 格式化 ──
    def step5():
        from app.generator import format_context, SYSTEM_PROMPT
        from langchain_core.documents import Document
        docs = [Document(page_content="测试内容", metadata={"filename": "test.txt"})]
        ctx = format_context(docs)
        assert "test.txt" in ctx
        assert "测试内容" in ctx
        print(f"  SYSTEM_PROMPT 长度: {len(SYSTEM_PROMPT)}")
        print(f"  格式化结果片段: {ctx[:80]}...")
    check("5. Prompt 格式化", step5)

    # ── 6. 分数提取 ──
    def step6():
        from app.utils import extract_score as _extract_score
        assert _extract_score("8") == 0.8
        assert _extract_score("8/10") == 0.8
        assert _extract_score("0.85") == 0.85
        assert _extract_score("分数：7") == 0.7
        print("  所有格式解析正确: '8'→0.8, '8/10'→0.8, '0.85'→0.85, '分数：7'→0.7")
    check("6. 分数提取", step6)

    # ── 7. Config 值校验（替代 ChromaDB 检索，无需 API Key）──
    def step7_config():
        """
        在 PR 阶段不调 ChromaDB/Embedding API，但必须校验：
        - embedding 模型名是否在已知列表中（换了名字会导致 init_db 失败）
        - top_k / chroma_collection_name / 模型名是否为合法值
        这些配置值错误会导致合并后 main 分支的 integration test 炸掉。
        """
        from app.config import settings

        # 已知有效的 embedding 模型名（阿里百炼平台支持的）
        KNOWN_EMBEDDING_MODELS = {
            "text-embedding-v3", "text-embedding-v2",
            "text-embedding-async-v2",
        }
        assert settings.embedding_model in KNOWN_EMBEDDING_MODELS, (
            f"EMBEDDING_MODEL='{settings.embedding_model}' 不在已知列表 {KNOWN_EMBEDDING_MODELS}，"
            "请确认 DashScope 支持该模型"
        )
        print(f"  embedding_model: {settings.embedding_model} (valid)")

        assert settings.top_k > 0, f"TOP_K={settings.top_k} 必须 > 0"
        print(f"  top_k: {settings.top_k} (valid)")

        assert isinstance(settings.chroma_collection_name, str) and settings.chroma_collection_name, (
            "CHROMA_COLLECTION_NAME 不能为空"
        )
        print(f"  chroma_collection_name: {settings.chroma_collection_name} (valid)")

        # 模型名不得为空
        for field in ("chat_model", "judge_model", "rewrite_model", "rerank_model"):
            val = getattr(settings, field, "")
            assert isinstance(val, str) and val, f"{field.upper()}='{val}' 不能为空"
        print(f"  chat/judge/rewrite/rerank models: all non-empty (valid)")

        print(f"  配置值校验通过（PR 模式，未调 API）")
    check("7. Config 值校验", step7_config)

    # ── 8. FastAPI app ──
    def step8():
        from app.main import app
        print(f"  title: {app.title}")
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
                if methods:
                    print(f"    {methods:6s} {route.path}")
    check("8. FastAPI app", step8)


def run_full_mode():
    """全量模式：直接调原始 verify.py（需 API Key + ChromaDB 已初始化）"""
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "verify.py")],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pr", "full"], default="pr")
    args = parser.parse_args()

    if args.mode == "full":
        run_full_mode()
    else:
        run_pr_mode()
        print(f"\n{'='*50}")
        print(f"  结果: {ok} 通过, {fail} 失败")
        print(f"{'='*50}")
        sys.exit(0 if fail == 0 else 1)
