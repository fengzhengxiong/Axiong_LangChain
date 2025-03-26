#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 13:44
# @Author  : fengzhengxiong
# @File    : document_processor.py

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from config_module import Config
from log_module import RAGLogger

class DocumentProcessor:
    def __init__(self):
        self.logger = RAGLogger().get_logger()
        # self.logger = logging.getLogger(__name__)
        self.splitter = self._init_splitter()
        # 优化建议：可以增加文件类型注册机制，支持更多格式

    def _init_splitter(self):
        """根据配置初始化文本分割器"""
        if Config.TEXT_SPLITTER_TYPE == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False
            )
        return CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        # 优化建议：可以增加语义分割器选项

    def load_documents(self) -> list:
        """加载并分割文档"""
        if not os.path.exists(Config.DATA_DIR):
            self.logger.warning(f"数据目录不存在: {Config.DATA_DIR}")
            return []

        try:
            loader = DirectoryLoader(
                Config.DATA_DIR,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True}
            )
            docs = loader.load()
            # 优化建议：可以增加文档预处理(如清洗、标准化)
            return self.splitter.split_documents(docs)
        except Exception as e:
            self.logger.error(f"文档加载失败: {str(e)}")
            return []
        # 优化建议：可以增加文档变更监测，增量更新