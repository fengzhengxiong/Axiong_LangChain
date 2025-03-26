#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 13:43
# @Author  : fengzhengxiong
# @File    : rag_system.py

import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from llm_model import LLMModel
from document_processor import DocumentProcessor
from retriever import HybridRetriever
from config_module import Config
from vector_store import VectorStore
from log_module import RAGLogger

class RAGSystem:
    """RAG系统核心"""
    # 优化建议：可以拆分为更小的组件

    def __init__(self):
        Config.validate()
        self.logger = RAGLogger().get_logger()

        # 确保向量库目录存在
        os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
        # 优化建议：可以增加目录权限检查

        try:
            # 初始化组件 - 优化建议：可以依赖注入
            self.llm = LLMModel()
            self.doc_processor = DocumentProcessor()

            # 加载文档
            documents = self.doc_processor.load_documents()
            if not documents:
                raise ValueError(f"未在 {Config.DATA_DIR} 中找到可处理的文档")
            # 优化建议：可以支持文档来源多样化

            # 初始化向量库
            self.vector_store = VectorStore()
            if not self.vector_store.store:
                self.logger.info("正在构建新的向量库...")
                if not self.vector_store.build(documents):
                    raise RuntimeError("向量库构建失败，请检查日志")
            # 优化建议：可以并行化处理

            # 初始化检索器
            self.retriever = HybridRetriever(documents).get_retriever()
            # 优化建议：可以支持检索器热切换

            self.qa_chain = self._create_qa_chain()
            self.logger.info("RAG系统初始化成功")
            # 优化建议：可以增加健康检查端点

        except Exception as e:
            self.logger.critical(f"系统初始化失败: {str(e)}", exc_info=True)
            raise RuntimeError("系统启动失败，请检查:\n"
                               f"1. 文档路径: {Config.DATA_DIR}\n"
                               f"2. 向量库路径: {Config.VECTOR_STORE_DIR}\n"
                               "3. 终端日志输出") from e

    def _create_qa_chain(self) -> RetrievalQA:
        """创建QA链"""
        prompt_template = """请基于以下上下文信息回答问题。如果无法回答，请说明原因。

        上下文：
        {context}

        问题：{question}

        请用中文给出专业回答："""
        # 优化建议：可以支持prompt模板管理

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm.model,
            chain_type="stuff",
            retriever=self.retriever,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="result"
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": Config.ENABLE_CACHE
            },
            return_source_documents=True
        )
        # 优化建议：可以支持多种chain类型

    def ask(self, query: str) -> str:
        """执行问答"""
        try:
            result = self.qa_chain.invoke({"query": query})
            # 优化建议：可以增加检索耗时统计

            # 记录检索结果
            if hasattr(result, 'source_documents'):
                self.logger.info(f"检索到 {len(result['source_documents'])} 条相关文档")
                # 优化建议：可以记录检索质量指标

            return result["result"]
        except Exception as e:
            self.logger.error(f"问答失败: {str(e)}")
            return "系统处理您的请求时出错"
            # 优化建议：可以细化错误类型

    def test_retrievers(self, query: str):
        """测试检索器"""
        documents = self.doc_processor.load_documents()
        print("\n=== 检索测试 ===")
        HybridRetriever.bm25_test(documents, query)

        vector_results = self.retriever.invoke(query)
        print("\n向量检索结果:")
        for i, doc in enumerate(vector_results):
            print(f"{i + 1}. Content: {doc.page_content[:100]}...")
        # 优化建议：可以增加混合检索结果对比