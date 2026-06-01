"""
========== 文档加载与分块模块 ==========
负责从文件读取文本，并将其切分成适合检索的片段（chunk）。
支持 TXT、MD、PDF 格式。

分块使用纯 Python 实现（避免 langchain_text_splitters 的 Rust tiktoken 扩展
在部分 Windows 环境下的 segfault 问题）。
"""

import os
from typing import List

from langchain_core.documents import Document as LCDocument

from app.config import settings
from app.exceptions import DocumentProcessingException


# ========== 纯 Python 递归文本切分器 ==========

class RecursiveTextSplitter:
    """
    纯 Python 实现的递归文本切分器，行为等价于
    langchain_text_splitters.RecursiveCharacterTextSplitter（length_function=len 模式）。
    避免 Rust tiktoken 原生扩展在部分 Windows 环境下的 segfault。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本"""
        final_chunks: List[str] = []

        # 找到第一个能把文本切开的 separator
        separator = separators[-1]  # 最细粒度
        for s in separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break

        # 按 separator 切分
        if separator == "":
            # 最细粒度：逐字符
            splits = list(text)
        else:
            splits = text.split(separator)

        # 合并过小的片段
        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                # 对超长片段递归
                if len(self.separators) > 1:
                    # 用剩余 separators 递归
                    next_index = self.separators.index(separator) + 1
                    if next_index < len(self.separators):
                        sub_chunks = self._split_text(s, self.separators[next_index:])
                        final_chunks.extend(sub_chunks)
                    else:
                        # 无更多 separator，硬切
                        final_chunks.append(s[:self.chunk_size])
                else:
                    final_chunks.append(s[:self.chunk_size])

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并片段，遵守 chunk_size 和 chunk_overlap"""
        docs: List[str] = []
        current_doc: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s)
            if current_len + s_len + (len(separator) if current_doc else 0) > self.chunk_size and current_doc:
                # 当前 chunk 满了
                doc_text = separator.join(current_doc)
                if doc_text.strip():
                    docs.append(doc_text)

                # 保留 overlap
                if self.chunk_overlap > 0:
                    # 从尾部拿 overlap 大小的内容
                    overlap_text = ""
                    for part in reversed(current_doc):
                        candidate = separator + part + overlap_text if overlap_text else part
                        if len(candidate) <= self.chunk_overlap:
                            overlap_text = candidate
                        else:
                            break
                    if overlap_text:
                        current_doc = [overlap_text]
                        current_len = len(overlap_text)
                    else:
                        current_doc = []
                        current_len = 0
                else:
                    current_doc = []
                    current_len = 0

            current_doc.append(s)
            current_len += s_len + (len(separator) if len(current_doc) > 1 else 0)

        # 最后一个 chunk
        if current_doc:
            doc_text = separator.join(current_doc)
            if doc_text.strip():
                docs.append(doc_text)

        return docs

    def split_text(self, text: str) -> List[str]:
        """入口：切分文本，返回字符串列表"""
        return self._split_text(text, list(self.separators))

    def create_documents(
        self, texts: List[str], metadatas: List[dict] = None
    ) -> List[LCDocument]:
        """创建 LangChain Document 对象"""
        documents: List[LCDocument] = []
        if metadatas is None:
            metadatas = [{}] * len(texts)
        for text, meta in zip(texts, metadatas):
            chunks = self.split_text(text)
            for chunk in chunks:
                documents.append(LCDocument(page_content=chunk, metadata=dict(meta)))
        return documents


def load_text_file(file_path: str) -> str:
    """
    读取文本文件的内容。
    支持 .txt, .md 格式。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """
    读取 PDF 文件的内容。
    使用 pypdf 库。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "读取 PDF 需要安装 pypdf 库：pip install pypdf"
        )
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n".join(texts)


def load_document(file_path: str) -> str:
    """
    根据文件扩展名自动选择加载方式。
    返回文档的纯文本内容。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf_file(file_path)
    elif ext in (".txt", ".md"):
        return load_text_file(file_path)
    else:
        raise DocumentProcessingException(
            message=f"不支持的文件格式：{ext}，仅支持 .txt, .md, .pdf",
            status_code=400,
        )


def split_document(text: str, filename: str) -> List[LCDocument]:
    """
    将长文本切分成多个片段（chunk），保留元数据。

    切分策略：
    - 按换行符、句号、空格等自然边界切分
    - 相邻 chunk 之间有重叠，避免切断关键信息
    - 每个 chunk 携带来源文件名和序号
    """
    splitter = RecursiveTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"filename": filename}],
    )

    # 给每个 chunk 标记序号
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def load_and_split(file_path: str) -> List[LCDocument]:
    """
    一键完成：加载文档 → 切分成片段。
    这是外部调用的主要接口。

    异常：
        DocumentProcessingException：文件读取失败或格式不支持
    """
    try:
        text = load_document(file_path)
        filename = os.path.basename(file_path)
        return split_document(text, filename)
    except DocumentProcessingException:
        raise
    except FileNotFoundError:
        raise DocumentProcessingException(
            message=f"文件不存在：{file_path}",
            detail="请确认文件路径是否正确",
            status_code=404,
        )
    except Exception as e:
        raise DocumentProcessingException(
            message="文档加载或分块失败",
            detail=f"{type(e).__name__}: {str(e)}",
        )
