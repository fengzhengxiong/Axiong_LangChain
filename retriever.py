#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/25 14:34
# @Author  : fengzhengxiong
# @File    : retriever.py

import warnings
from typing import List
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from vector_store import VectorStore
from config_module import Config

class HybridRetriever:
    """混合检索器实现"""
    # 优化建议：可以增加检索器抽象基类

    def __init__(self, documents: List[Document]):
        # 忽略BM25库的警告信息
        warnings.filterwarnings("ignore", module="rank_bm25")

        # 向量检索
        self.vector_store = VectorStore()
        self.vector_retriever = self.vector_store.get_retriever()
        # 优化建议：可以延迟初始化

        # BM25检索
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k1=Config.BM25_K1,
            b=Config.BM25_B
        )
        self.bm25_retriever.k = Config.TOP_K
        # 优化建议：可以增加BM25参数调优接口

    def get_retriever(self):
        """获取当前检索器"""
        if Config.RETRIEVER_STRATEGY == "hybrid":
            return EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[Config.BM25_WEIGHT, Config.VECTOR_WEIGHT]
            )
        elif Config.RETRIEVER_STRATEGY == "bm25":
            return self.bm25_retriever
        return self.vector_retriever
        # 优化建议：可以支持自定义权重动态调整

    @staticmethod
    def bm25_test(documents: List[Document], query: str, k: int = 3):
        """增强版BM25检索测试"""
        try:
            retriever = BM25Retriever.from_documents(documents)
            results = retriever.invoke(query, top_k=k)

            print("\n=== BM25检索测试 ===")
            print(f"查询: '{query}'")
            print(f"返回结果数: {len(results)}")

            if not results:
                print("⚠️ 未检索到任何结果")
                return

            for i, doc in enumerate(results):
                score = doc.metadata.get('score', 'N/A')

                # 安全处理score显示
                try:
                    score_str = f"{float(score):.3f}" if not isinstance(score, str) else score
                except (ValueError, TypeError):
                    score_str = str(score)

                print(f"\n结果 {i + 1}:")
                print(f"Score: {score_str}")
                print(f"来源: {doc.metadata.get('source', '未知')}")
                print(f"内容预览: {doc.page_content[:100]}{'...' if len(doc.page_content) > 100 else ''}")

        except Exception as e:
            print(f"BM25测试失败: {str(e)}")
            raise