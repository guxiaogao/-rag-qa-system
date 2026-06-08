"""
========== 初始化向量数据库 ==========
首次运行时执行：
1. 写入内置示例文档到 data/source_docs/（零配置跑通）
2. 自动扫描 data/source_docs/ 目录下所有 .txt/.md/.pdf 文件
3. 逐文件加载、分块并写入 ChromaDB

运行方式：
  python scripts/init_db.py          # 增量模式：只索引新文件
  python scripts/init_db.py --full   # 全量模式：清空旧数据后重建

说明：
- 增量模式（默认）：比较文件名，跳过已入库的文件，只处理新文件。
  如需更新已索引文件的内容，请用 --full 重建。
- 全量模式（--full）：清空向量库后重新索引所有文件。
- 将你的 PDF/论文/文档放入 data/source_docs/，运行此脚本即可自动索引。
"""

import os
import sys

# 确保可以从项目根目录导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.document_loader import load_and_split
from app.database import get_vector_store, reset_collection, get_indexed_filenames
from app.config import settings, PROJECT_ROOT


# ========== 内置示例文档 ==========
# 为保证零配置即可跑通，内置 1 篇示例文档（简介.txt）。
# 首次运行时会写入 data/source_docs/，后续可自行增删文件。
# 放入该目录的任何 .txt / .md / .pdf 都会被自动索引。

SAMPLE_DOCUMENTS = {
    "简介.txt": """“北师珠”通常指“北京师范大学珠海分校”，这是一所有着特殊历史轨迹的大学。

简单来说，作为独立学院的“北京师范大学珠海分校”（北师珠）已于2021年停止招生，其办学使命已于2024年正式结束，现已成为历史。当前，在其原址上运作的是“北京师范大学珠海校区”，这是北师大“双一流”建设的重要组成部分。

为了方便理解，我将两者的发展沿革整理如下：

时间节点	实体名称	关键事件与性质
2001年	北京师范大学珠海教育园区	奠基成立，作为北师大高等教育改革的试验园区。
2003年	北京师范大学珠海分校 (北师珠)	经教育部批准，更名为珠海分校，成为一所由北师大和珠海市政府合作举办的独立学院。
2017年	北师珠/珠海校区（规划）	广东省、珠海市与北师大签署协议，决定共建“北京师范大学珠海校区”。
2019年	北京师范大学珠海分校 / 北京师范大学珠海校区	教育部正式批复同意建设北京师范大学珠海校区，并要求珠海分校逐年调减招生计划。
2021年	北京师范大学珠海分校	作为独立学院的珠海分校正式停止招生。
2024年	北京师范大学珠海分校	作为独立学院的珠海分校终止办学。
🎓 “北师珠”：一所特殊的“中国高校锦鲤”
在2019-2024年期间，北师珠（分校）因其即将“华丽转身”为985校区而备受瞩目，被称为“中国高校最强锦鲤”。

从二本到985：北师珠的录取分数线经历了快速攀升。2015年，其在广东的招生批次调整为二本A类；到2018年，已在四川、青海等省份实现一本招生。随着其向珠海校区的转型，其未来将共享北师大985/211和“双一流”的资源与声誉。

性质与毕业证：需要明确的是，“分校”与“校区”有本质区别。在2019年及之前以珠海分校名义入学的学生，毕业时获得的是“北京师范大学珠海分校”的毕业证书。而在2020年及以后以珠海校区名义招录的学生，获得的将是与北京校区一致的“北京师范大学”毕业证书和学位证书。

🏫 新生事物：“北京师范大学珠海校区”
作为接棒者的北京师范大学珠海校区，代表了更高的办学层次。它在北师大的“一体两翼”战略中，是与北京校区地位相同、一体规划的南方校区。

学科实力：珠海校区拥有北师大作为“双一流”建设高校的核心学科资源，16个A类学科已在此落户。学校聚焦教师教育、脑科学、生态环境、经济管理等六大领域。

培养模式：珠海校区目前提供19个本科专业招生，并创新性地实行“学院+书院”的协同育人模式，将专业学习与综合素质培养相结合。

优美校园：校区所在地占地5000余亩，三面环山、一面向海，被誉为“亚洲最美丽的山谷大学”。校园无线Wi-Fi全覆盖，具备得天独厚的自然风光和现代化的学习生活环境。

💎 总结
总而言之，虽然“北师珠”（作为分校）的名字已经成为历史，但它开启的北京师范大学珠海校区，正以全新的姿态和更高的标准，承担起培养人才的重要使命。

"""
}


def main():
    """初始化向量数据库：写入示例文档 + 自动扫描并索引 source_docs 目录"""
    # 解析命令行参数
    full_rebuild = "--full" in sys.argv

    mode_label = "全量重建" if full_rebuild else "增量索引（跳过已入库文件）"
    print("=" * 60)
    print(f"初始化向量数据库 — {mode_label}")
    print("=" * 60)

    # ========== 第 0 步：检查 API Key ==========
    if not settings.dashscope_api_key or settings.dashscope_api_key == "your_api_key_here":
        print("\n[错误] 未配置 DASHSCOPE_API_KEY")
        print("   请将 .env.example 复制为 .env 文件，填入你的 API Key")
        print("   API Key 申请地址：https://dashscope.aliyun.com/")
        return

    # ========== 第 1 步：确保目录存在并写入内置示例文档 ==========
    data_dir = os.path.join(PROJECT_ROOT, "data", "source_docs")
    os.makedirs(data_dir, exist_ok=True)

    print("\n[第 1 步] 写入内置示例文档...")
    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = os.path.join(data_dir, filename)
        # 写入文件（如果用户已手动放了同名文件则覆盖，保证内容最新）
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   [OK] {filename}")

    # ========== 第 2 步：扫描 source_docs 目录，收集所有支持的文件 ==========
    # 自动发现用户放入的 PDF、额外 txt 等，无需手动配置
    print(f"\n[第 2 步] 扫描 {data_dir} 目录...")

    # 支持的文件扩展名（与 document_loader.py 中的 load_document 保持一致）
    SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")

    # os.listdir() 列出目录下所有文件
    # 过滤：只保留扩展名在支持列表中的文件，跳过子目录
    all_files = os.listdir(data_dir)
    all_supported = []
    for f in all_files:
        full_path = os.path.join(data_dir, f)
        # 跳过子目录，只处理文件
        if not os.path.isfile(full_path):
            continue
        # 检查扩展名（转小写后比较，兼容 .PDF / .TXT 等大写写法）
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            all_supported.append(f)
        else:
            print(f"   [SKIP] 跳过（不支持格式）：{f}")

    if not all_supported:
        print("\n[错误] 未找到任何可处理的文档文件（.txt/.md/.pdf）")
        return

    all_supported.sort()

    # ========== 第 3 步：确定哪些文件需要索引 ==========
    if full_rebuild:
        # 全量模式：清空旧数据后重新索引全部文件
        print(f"\n[第 3 步] 全量模式：清空向量数据库中的旧数据...")
        try:
            reset_collection()
            print(f"   已删除旧集合：{settings.chroma_collection_name}")
        except Exception as e:
            print(f"   清理旧数据时出错（可忽略）：{e}")
        files_to_process = all_supported
    else:
        # 增量模式：查询已入库的文件名，找出新文件
        print(f"\n[第 3 步] 增量模式：比对已索引文件...")
        try:
            indexed = get_indexed_filenames()
            print(f"   向量库中已有 {len(indexed)} 个文件")
        except Exception as e:
            print(f"   查询向量库失败 ({e})，将当作首次运行处理全部文件")
            indexed = set()

        files_to_process = []
        for f in all_supported:
            if f in indexed:
                print(f"   [SKIP] 已入库：{f}")
            else:
                files_to_process.append(f)

    if not files_to_process:
        print(f"\n{'=' * 60}")
        print(f"没有新文件需要索引，向量库已是最新。")
        print(f"   已索引文件数：{len(all_supported)}")
        print(f"   如需强制重建，请运行：python scripts/init_db.py --full")
        return

    print(f"   发现 {len(files_to_process)} 个新文件待索引：")
    for f in files_to_process:
        size_kb = os.path.getsize(os.path.join(data_dir, f)) / 1024
        print(f"     - {f}（{size_kb:.1f} KB）")

    # ========== 第 4 步：逐文件加载、分块、写入向量库 ==========
    print(f"\n[第 4 步] 加载文档并写入向量数据库...")

    # 提前创建向量库实例（复用同一连接，比循环内每次创建更高效）
    vector_store = get_vector_store()
    total_chunks = 0

    for filename in files_to_process:
        file_path = os.path.join(data_dir, filename)
        print(f"\n  -- 处理：{filename} --")

        # load_and_split() 内部自动根据扩展名选择解析方式：
        #   .txt/.md → 直接读取文本
        #   .pdf     → pypdf 提取文本
        chunks = load_and_split(file_path)
        chunk_count = len(chunks)
        print(f"     切分：{chunk_count} 个片段")

        # add_documents() 会调用 Embedding API 将每个片段向量化后写入 ChromaDB
        vector_store.add_documents(chunks)
        total_chunks += chunk_count
        print(f"     已写入向量库")

    # ========== 第 5 步：输出汇总 ==========
    print(f"\n{'=' * 60}")
    print(f"初始化完成！")
    print(f"   本次新增文件数：{len(files_to_process)}")
    print(f"   本次新增片段数（chunks）：{total_chunks}")
    print(f"   向量数据库：{settings.chroma_db_path}")
    print(f"   文档目录：{data_dir}")
    print(f"\n后续使用：")
    print(f"   启动服务：uvicorn app.main:app --reload")
    print(f"   API 文档：http://localhost:8000/docs")
    print(f"   聊天界面：http://localhost:8000")
    print(f"\n[提示] 将新的 PDF/txt/md 文档放入 source_docs/ 后，")
    print(f"   运行 python scripts/init_db.py 增量追加新文件")
    print(f"   运行 python scripts/init_db.py --full 重建全部索引")


if __name__ == "__main__":
    main()
