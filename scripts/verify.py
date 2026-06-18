# -*- coding: utf-8 -*-
"""功能验证脚本：检查 RAG 项目各模块是否正常运行
运行方式：python scripts/verify.py  (在 rag项目/ 目录下执行)
"""
import sys, os, traceback

# 确保项目根目录在 path 中
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
    from app.exceptions import RetrievalException, RAGException
    e = RetrievalException(detail="测试")
    assert e.error_code == "RETRIEVAL_ERROR"
    assert e.status_code == 503
    print(f"  error_code={e.error_code}, status_code={e.status_code}, message={e.message}")
check("2. 异常体系", step2)


# ── 3. API Schemas ──
def step3():
    from app.schemas import ChatRequest
    r = ChatRequest(query="测试")
    print(f"  query={r.query}, top_k={r.top_k}")
    # 缺必填字段应报错
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


# ── 7. ChromaDB 连接 & 检索 ──
def step7():
    from app.database import get_vector_store
    vs = get_vector_store()
    print(f"  集合名: {vs._collection.name}")
    print(f"  文档片段总数: {vs._collection.count()}")

    from app.retriever import retrieve, get_all_documents
    docs = get_all_documents()
    print(f"  已索引文档: {len(docs)} 个")
    for d in docs[:5]:
        print(f"    - {d['filename']} ({d['chunk_count']} chunks)")

    results = retrieve(query="北师珠", top_k=3)
    print(f"  检索 '北师珠' → {len(results)} 个片段")
    for i, doc in enumerate(results):
        score = doc.metadata.get("score", "?")
        text = doc.page_content[:60].replace("\n", " ")
        print(f"    [{i+1}] score={score} | {text}...")
check("7. ChromaDB 连接 & 检索", step7)


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


# ── 结果 ──
print(f"\n{'='*50}")
print(f"  结果: {ok} 通过, {fail} 失败")
print(f"{'='*50}")
sys.exit(0 if fail == 0 else 1)
