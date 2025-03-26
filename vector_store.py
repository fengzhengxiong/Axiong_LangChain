#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 13:44
# @Author  : fengzhengxiong
# @File    : vector_store.py

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from config_module import Config
from log_module import RAGLogger


class VectorStore:
    """健壮的FAISS向量存储管理"""
    # 优化建议：可以抽象为接口，支持多种向量库

    def __init__(self):
        # self.logger = logging.getLogger(__name__)
        self.logger = RAGLogger().get_logger()
        # 显式设置环境变量确保使用CPU
        os.environ['FAISS_NO_GPU'] = '1'
        self.embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        self.store = self._init_vector_store()
        # 优化建议：可以增加embedding模型健康检查

    def _init_vector_store(self) -> Optional[FAISS]:
        """安全初始化向量存储"""
        try:
            # 使用Path对象确保路径安全
            store_path = Path(Config.VECTOR_STORE_DIR)
            if store_path.exists():
                required_files = ['index.faiss', 'index.pkl']
                if all((store_path / f).exists() for f in required_files):
                    self.logger.info(f"从 {store_path} 加载已有向量库")
                    return FAISS.load_local(
                        folder_path=str(store_path),  # 显式转换为字符串
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=Config.ALLOW_DANGEROUS_DESERIALIZATION
                    )
                self.logger.warning("向量库文件不完整")
            return None
        except Exception as e:
            self.logger.error(f"加载向量库失败: {str(e)}", exc_info=True)
            return None
        # 优化建议：可以增加版本兼容性检查

    def build(self, documents: List[Document]) -> bool:
        """构建新的向量库"""
        try:
            store_path = Path(Config.VECTOR_STORE_DIR)
            store_path.mkdir(parents=True, exist_ok=True)

            self.logger.info("开始构建向量库...")
            self.store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # 优化建议：可以增加构建进度显示

            self.store.save_local(folder_path=str(store_path))
            self.logger.info(f"向量库构建完成，保存到 {store_path}")
            return True
        except Exception as e:
            self.logger.error(f"构建向量库失败: {str(e)}", exc_info=True)
            return False
        # 优化建议：可以支持增量构建

    def get_retriever(self):
        """获取安全的检索器"""
        if not self.store:
            raise ValueError("向量库未初始化，请检查：\n"
                             "1. data目录是否有文档\n"
                             "2. vector_store目录权限\n"
                             "3. 磁盘空间是否充足")

        search_kwargs = {"k": Config.TOP_K}
        if Config.SEARCH_TYPE == "similarity_score_threshold":
            search_kwargs["score_threshold"] = Config.SEARCH_SCORE_THRESHOLD

        return self.store.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
        # 优化建议：可以支持自定义相似度计算方式