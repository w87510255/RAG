# ====================== 导入所有依赖 ======================
import os
import re
import sys

import pandas as pd
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
# ========== 高精度 Qdrant 配置 ==========
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from typing import List,Any
from langchain_core.documents import Document
#按照语义拆分向量数据库使用的文档数据
from langchain_experimental.text_splitter import SemanticChunker
import openpyxl
#自定义
from excel_head import detect_header_final_v2

SHOW_SOURCE_AND_PARAGRAPH = True  # 👈 在这里改：True=显示 | False=不显示
SHOW_EXCEL_DATA = False

answer_cache = []      # 缓存检索到的文档（最多3条）
current_answer_idx = 0  # 当前显示第几个答案
last_user_question = "" # 记录上一个问题，防止切换问题错乱
synonym_dict = {}

# 文档加载器
from langchain_community.document_loaders import (
    TextLoader,        # TXT
    PyPDFLoader,      # PDF
    Docx2txtLoader,       # Word (.docx)
    UnstructuredExcelLoader #Excel
)

# ====================== 配置本地大模型 Qwen (vLLM) ======================
llm = ChatOpenAI(model="Qwen3.6-35B-A3B", base_url="http://10.1.100.94:8000/v1",api_key="none", temperature=0)

# ====================== 本地 Embedding 模型 ======================
embedding = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://192.168.29.10:11434/"
)

def load_synonym_dict(excel_path="./业务同义词词典.xlsx"):
    """
    读取Excel同义词表，加载到全局synonym_dict
    异常处理：文件不存在/读取失败，不影响原有代码运行
    """
    global synonym_dict
    if not os.path.exists(excel_path):
        print(f"⚠️  未找到同义词词典文件：{excel_path}，跳过同义词扩展")
        return

    try:
        df = pd.read_excel(excel_path, sheet_name="同义词表")
        # 校验列名
        if "核心业务词" not in df.columns or "关联同义词/词组" not in df.columns:
            print("⚠️  同义词表列名错误，必须包含【核心业务词】和【关联同义词/词组】")
            return

        # 构建同义词词典
        for _, row in df.iterrows():
            core_word = str(row["核心业务词"]).strip()
            synonym_str = str(row["关联同义词/词组"]).strip()
            if not core_word or synonym_str == "nan":
                continue
            # 拆分同义词，去重
            synonyms = list(set([w.strip() for w in synonym_str.split(",") if w.strip()]))
            synonym_dict[core_word] = synonyms

        print(f"✅ 成功加载业务同义词词典，共{len(synonym_dict)}个核心词")
    except Exception as e:
        print(f"⚠️  同义词词典读取失败：{e}，跳过同义词扩展")

# ====================== 新增：用同义词扩展用户问题 ======================
def expand_question_with_synonym(question):
    """
    把用户问题里的核心词，替换成「核心词+所有同义词」，提升检索召回率
    例：用户问"乱码" → 扩展为"乱码 编码 字符集 GBK UTF-8 中文乱码 字符异常"
    """
    if not synonym_dict:
        return question

    expanded_words = []
    question_lower = question.strip().lower()

    # 先把原问题的词加进去
    expanded_words.extend(question_lower.split())

    # 匹配核心词，添加同义词
    for core_word, synonyms in synonym_dict.items():
        if core_word.lower() in question_lower:
            expanded_words.extend(synonyms)

    # 去重，拼接成最终的扩展问题
    expanded_words = list(set(expanded_words))
    expanded_question = " ".join(expanded_words)

    print(f"🔍 原问题：{question}")
    print(f"🔍 扩展后问题：{expanded_question}")
    return expanded_question

# ====================== 自动加载文件夹里所有 PDF/DOCX/TXT ======================
def load_all_docs(folder_path: str = "./docs"):
    docs = []
    # 自动创建 docs 文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 遍历所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # TXT
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        # PDF
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        # WORD (.docx)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith((".xlsx", ".xls")):
            loaded_docs = load_excel_smart_simple(file_path)
            if SHOW_EXCEL_DATA:
                print("excel:",loaded_docs)
        else:
            continue

        if not filename.endswith((".xlsx", ".xls")):
            loaded_docs = loader.load()

        for doc in loaded_docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = filename
            doc.metadata["filename"] = filename
            if filename.endswith((".xlsx", ".xls")):
                doc.metadata["file_type"] = "excel"
        #加载文档
        docs.extend(loaded_docs)

    return docs

def load_excel_smart_simple(file_path, max_rows_per_sheet=30, max_columns=15):
    """
    智能但简单的Excel加载器
    核心思想：表头+数据，让LLM自己理解
    """
    try:
        docs = []
        filename = os.path.basename(file_path)
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            try:
                head_num = detect_header_final_v2(file_path)
                print(file_path,"headnum:",head_num)
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=head_num)
                if df.empty:
                    print(f"  ⏭️  Sheet为空: {sheet_name}")
                    continue
                print(f"📊 处理: {sheet_name} ({len(df)}行×{len(df.columns)}列)")
                # 1. 如果表格太大，分块处理
                if len(df) > max_rows_per_sheet or len(df.columns) > max_columns:
                    sheet_docs = split_large_table(df, filename, sheet_name, max_rows_per_sheet, max_columns)
                else:
                    # 2. 小表格直接处理
                    sheet_docs = [create_table_doc(df, filename, sheet_name)]
                docs.extend(sheet_docs)

            except Exception as e:
                print(f"    ⚠️  Sheet {sheet_name} 失败: {e}")
                continue
        return docs
    except Exception as e:
        print(f"❌ Excel读取失败: {e}")
        return []

def create_table_doc(df, filename, sheet_name, is_chunk=False, chunk_info=None):
    """创建表格文档（核心函数）"""
    # 限制列数，避免太长
    display_cols = min(12, len(df.columns))
    # 构建内容
    content_parts = []
    # 1. 标题
    if chunk_info:
        content_parts.append(f"【表格块: {sheet_name} ({chunk_info})】")
    else:
        content_parts.append(f"【表格: {sheet_name}】")

    # 2. 表头
    columns = [str(col) for col in df.columns[:display_cols]]
    content_parts.append(f"表头: {', '.join(columns)}")

    if len(df.columns) > display_cols:
        content_parts.append(f"(还有{len(df.columns)-display_cols}列未显示)")

    # 3. 数据行（限制行数）
    display_rows = min(25, len(df))
    for i in range(display_rows):
        row = df.iloc[i]
        row_values = []

        for col in df.columns[:display_cols]:
            val = row[col]
            if pd.isna(val):
                row_values.append("空")
            else:
                # 简化长文本
                val_str = str(val)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                row_values.append(val_str)

        content_parts.append(f"行{i+1}: {' | '.join(row_values)}")
    # 4. 统计信息
    if len(df) > display_rows:
        content_parts.append(f"... 还有 {len(df)-display_rows} 行")

    content = "\n".join(content_parts)
    # 元数据
    metadata = {
        "source": filename,
        "filename": filename,
        "sheet_name": sheet_name,
        "file_type": "excel",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "display_rows": display_rows,
        "display_columns": display_cols
    }

    if chunk_info:
        metadata["chunk_info"] = chunk_info
        metadata["doc_type"] = "table_chunk"
    else:
        metadata["doc_type"] = "table"

    return Document(page_content=content, metadata=metadata)

def split_large_table(df, filename, sheet_name, max_rows=30, max_cols=12):
    """分割大表格"""
    docs = []
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. 先按行分块
    for row_start in range(0, total_rows, max_rows):
        row_end = min(row_start + max_rows, total_rows)
        row_chunk = df.iloc[row_start:row_end]

        # 2. 如果列太多，再按列分块
        if total_cols <= max_cols:
            # 列不多，直接创建文档
            chunk_info = f"行{row_start+1}-{row_end}"
            doc = create_table_doc(row_chunk, filename, sheet_name, True, chunk_info)
            docs.append(doc)
        else:
            # 列太多，按列分块
            for col_start in range(0, total_cols, max_cols):
                col_end = min(col_start + max_cols, total_cols)
                col_chunk = row_chunk.iloc[:, col_start:col_end]

                chunk_info = f"行{row_start+1}-{row_end}, 列{col_start+1}-{col_end}"
                doc = create_table_doc(col_chunk, filename, sheet_name, True, chunk_info)
                docs.append(doc)

    return docs

def load_single_doc(filename, folder_path="./docs"):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        return []
    return loader.load()

def split_excel_document(content, metadata, max_chunk_size=400):
    """专门处理 Excel 文档分块"""
    if not content:
        return []

    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Excel 特有的逻辑：以"行X:"为自然分界
        if line.startswith("行") and ":" in line:
            if current_chunk and current_length >= max_chunk_size * 0.7:
                # 保存当前块
                chunk_text = '\n'.join(current_chunk)
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=metadata.copy()
                )
                chunks.append(chunk_doc)
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line)
        else:
            current_chunk.append(line)
            current_length += len(line)

        # 如果块太大，切割
        if current_length >= max_chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=metadata.copy()
            )
            chunks.append(chunk_doc)
            current_chunk = []
            current_length = 0

    # 处理最后一块
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunk_doc = Document(
            page_content=chunk_text,
            metadata=metadata.copy()
        )
        chunks.append(chunk_doc)

    return chunks

# ====================== 文档分块 ======================
def split_documents(docs, breakpoint_threshold = 90):
    if not docs:
        raise ValueError("没有加载到任何文档！请在docs文件夹放入文件")

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=512,
    #     chunk_overlap=20,
    #     separators=["\n\n", "\n", "。", "，", " "]
    # )
    # return splitter.split_documents(docs)
    all_splits = []
    for i, doc_item in enumerate(docs, 1):
        metadata = doc_item.metadata.copy()
        file_type = metadata.get("file_type", "unknown")
        if file_type == "excel":
                #content = doc_item.page_content
                # Excel 文档用不同策略
                #excel_splits = split_excel_document(content, metadata)
                all_splits.append(doc_item)
        else:
            # 按语义拆分
            semantic_splitter = SemanticChunker(
                embeddings=embedding,  # 你的 OllamaEmbeddings
                breakpoint_threshold_type="percentile",  # 或 "standard_deviation"
                breakpoint_threshold_amount=breakpoint_threshold,  # 85-95
            )
            splits = semantic_splitter.split_documents([doc_item])
            all_splits.extend(splits)

    print(f"✅ 语义分块完成: {len(docs)} 文档 → {len(all_splits)} 分块")
    return all_splits

def get_docs_file_list(folder_path="./docs"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    file_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".pdf", ".docx",".xlsx", ".xls")):
            file_list.append(filename)
    return file_list

def select_document():
    #folder = "./docs"
    #file_list = get_docs_file_list(folder)
    #if not file_list:
    #    print("❌ docs 文件夹下没有找到 txt/pdf/docx 文档")
    #    return None
    return load_all_docs()
    # print("\n===== 文档列表 =====")
    # for idx, name in enumerate(file_list):
    #     print(f"{idx+1}. {name}")
    # print(f"{len(file_list)+1}. 加载全部文档")
    #
    # while True:
    #     try:
    #         choice = int(input("\n请输入序号选择文档："))
    #         if 1 <= choice <= len(file_list):
    #             selected_name = file_list[choice-1]
    #             print(f"✅ 已选择：{selected_name}")
    #             return load_single_doc(selected_name)
    #         elif choice == len(file_list)+1:
    #             print("✅ 已选择：加载全部文档")
    #             return load_all_docs()
    #         else:
    #             print("❌ 输入序号超出范围，请重新输入")
    #     except ValueError:
    #         print("❌ 请输入数字")


# ====================== 构建向量库 ======================
def build_retrievers():
    force_rebuild = True
    persist_path = "./qdrant_rag_db"
    # 清空旧库（等价于原来删除 chroma_rag_db）
    import shutil
    if force_rebuild and os.path.exists("./qdrant_rag_db"):
        print("🧹 删除旧向量数据库...")
        shutil.rmtree("./qdrant_rag_db")

    docs = load_all_docs()
    splits = split_documents(docs)
    # 1. 语义检索
    vector_db = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embedding,
        path=persist_path,        # 只传这个！
        collection_name="rag_db",
        force_recreate=force_rebuild,
         # ====================== 🔥 两个提升精度 30% 的参数 ======================
        distance=models.Distance.COSINE,          # 1. 余弦相似度（业务RAG必开）
        payload_dense_index=True,                 # 2. 密集索引（排序更准）
        # ======================================================================
    )

    semantic_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. BM25 关键词检索（你要保留的）
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3
    bm25_retriever.docs = splits

    return semantic_retriever, bm25_retriever


def Bm25EerorCode(question, bm25_retriever):
    code_match = re.search(r'-?\d{2,5}', question)
    error_code = code_match.group() if code_match else None
    docs_code = []
    if error_code:
        print(f"✅ 检测到错误码：{error_code} → 强制检索匹配该错误码的文档")

        # 从 BM25 的全部文档里 强制找包含错误码的段落
        all_docs = bm25_retriever.docs  # 全部文档块
        for doc in all_docs:
            if error_code in doc.page_content:  # 完整匹配 -108
                docs_code.append(doc)
    return docs_code


def show_all_cache():
    """
    遍历并打印 answer_cache 里所有缓存的文档内容
    """
    global answer_cache
    if not answer_cache:
        print("⚠️ answer_cache 暂无缓存内容")
        return

    print(f"\n===== 📦 全部缓存答案（共 {len(answer_cache)} 条）=====")
    for i, doc in enumerate(answer_cache):
        print(f"\n【第 {i+1} 条缓存】")
        print(f"来源文件：{doc.metadata.get('source', '未知文件')}")
        print(f"内容：{doc.page_content[:150]}...")  # 只打印前150字符避免刷屏
    print("===================================================\n")

def  hybrid_retrieve(question, semantic_retriever, bm25_retriever):
    """
    自定义混合检索：
    1. 语义检索
    2. BM25 关键词检索
    3. 合并结果 + 去重
    完全不依赖 EnsembleRetriever！
    """
    expanded_question = expand_question_with_synonym(question)

    # 1. 两路检索（用扩展后的问题检索，大幅提升召回率）
    docs_semantic = semantic_retriever.invoke(expanded_question)
    docs_bm25 = bm25_retriever.invoke(expanded_question)
    # 1. 两路检索
    #docs_semantic = semantic_retriever.invoke(question)
    #docs_bm25 = bm25_retriever.invoke(question)

    # 2. 合并
    combined = docs_semantic + docs_bm25 + Bm25EerorCode(question, bm25_retriever)
    #combined = docs_semantic + docs_bm25
    # 3. 去重（按内容）
    unique_docs = []
    content_set = set()
    for doc in combined:
        if doc.page_content not in content_set:
            content_set.add(doc.page_content)
          #  print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
          #  print("ret-doc:", doc.page_content)
          #  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            unique_docs.append(doc)

    print(f"🔍 混合检索返回 {len(unique_docs)} 条结果")
    return unique_docs
# ================== ==== 构建 RAG 链 ======================
def create_rag_chain():
    print("开始产生RAG")
    load_synonym_dict()

    docs = select_document()
    if not docs:
        return None
    print("开始产生向量数据库")
    #retriever = build_hybrid_retriever()
    semantic_ret, bm25_ret = build_retrievers()

    prompt = ChatPromptTemplate.from_template("""
    你是一个数据同步公司的问答助手,该公司软件最重要功能是建立规则同步全量数据,和增量数据,规则异常就是数据同步异常.
    你是文档问答助手,请根据下面的文档内容回答问题,不要编造答案.
    注意：文档中可能包含Excel表格数据，格式通常为：
    [表格: 名称]表头: 列1, 列2, 列3
    行1: 值1 | 值2 | 值3
    行2: 值1 | 值2 | 值3
    如果没有找到符合问题答案,提示让问题变的更详细些,如果直接说某个模块有什么报错#
    文档内容的文件名：
    文档内容：{context}
    用户问题：{question}
    回答：...
    参考来源：文件名 | 原文段落  
    不同的参考来源直接添加换行符
    响应内容引用不同的段落之间添加换行符
    """)

    def format_docs(docs):
        if not docs:
            return ""
        if SHOW_SOURCE_AND_PARAGRAPH:
            formatted = []
            for doc in docs:
                filename = doc.metadata.get("source", "未知文件")
                content = doc.page_content
                formatted.append(f"[来自文件：{filename}]\n{content}")
            return "\n\n".join(formatted)
        else:
            return "\n\n".join(doc.page_content for doc in docs)

    def retrieve(question):
        global answer_cache, current_answer_idx, last_user_question

        # 如果用户输入“下一个”，切换答案**
        if question.strip() == "下一个":
            if len(answer_cache) == 0:
                print("⚠️ 暂无缓存答案，请先提问！")
                return []
            # 最多切换3次，到第3个后停止**
            if current_answer_idx < len(answer_cache) - 1:
                current_answer_idx += 1
                print(f"✅ 显示第 {current_answer_idx+1} 个答案")
            else:
                print("⚠️ 已到最后一个答案，共3个！")
            return [answer_cache[current_answer_idx]]

        # 正常提问：重新检索，缓存最多前3条**
        docs = hybrid_retrieve(question, semantic_ret, bm25_ret)
        last_user_question = question
        answer_cache = docs[:3]  # 只保存最多3条**
        show_all_cache()
        current_answer_idx = 0 # 重置为第一个**
        return docs
        # if len(answer_cache) == 0:
        #     return []  # 默认返回第一条**
        # else:
        #     return [answer_cache[0]]

    #"context":retrieve
    rag_chain = (
        {"context": RunnableLambda(retrieve) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("产生rag")
    return rag_chain


def TestLLM():
    message = [SystemMessage(content="你是一个数据容灾，同步公司的内部助手，当用户要求创建规则时，你必须调用CreateRule工具."),
           HumanMessage(f"你知道什么是全量和增量吗？")]
    ret = llm.invoke(message)
    print(ret)

# ====================== 测试运行 ======================
if __name__ == "__main__":
    #TestLLM()

    # excelfile = "./docs/DM-DM备库抽取大数据量同步测试.xlsx"
    # excelfile2 = "./docs/11469_DM备库抽取基本功能测试报告20240201.xlsx"
    # head_num = detect_header_final_v2(excelfile2)
    # print(head_num)
    # df = pd.read_excel(excelfile2, header=head_num)
    # print(df.head())
    # sys.exit()


    rag_chain = create_rag_chain()

    #question1 = "增量没数据是什么原因.请根据文档内容回答问题"
    #question1 = "你这个文档说的是什么.请根据文档内容回答问题"
    #question2 = "规则正常但是数据没同步是什么原因"
    #question2 = "达梦同步报错-108是什么原因"
    #result = rag_chain.invoke(question2)
    #print(result)

    while True:
        question = input("\n!!!提问少说废话,请将'iatrack报错-108的原因是什么'改成'iatrack报错-108',只说关键信息.请输入问题:")

        if question.strip().lower() == "exit":
            break

        rep = rag_chain.invoke(question)
        print("question:",question,"\n")
        print("response:",rep)