#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 11:59
# @Author  : fengzhengxiong
# @File    : config.py

from typing import ClassVar, Optional

class Config:
        # 路径配置
        DATA_DIR: ClassVar[str] = "data"                               # 建议：可以支持环境变量覆盖
        VECTOR_STORE_DIR: ClassVar[str] = "vector_store"               # 建议：增加索引版本管理

        # 模型配置
        LLM_MODEL: ClassVar[str] = "deepseek-r1:8b"                    # 建议：支持模型切换时的自动适配
        EMBEDDING_MODEL: ClassVar[str] = "nomic-embed-text:latest"     # 建议：增加embedding维度配置
        LLM_TEMPERATURE: ClassVar[float] = 0.3                         # 合理范围
        LLM_MAX_TOKENS: ClassVar[int] = 2048                           # 建议：根据模型调整

        # 文本处理
        CHUNK_SIZE: ClassVar[int] = 500                                # 文本分割的文本块大小
        CHUNK_OVERLAP: ClassVar[int] = 50                              # 文本分割的文本块之间的重叠大小
        TEXT_SPLITTER_TYPE: ClassVar[str] = "recursive"                # 文本分割方式 1:基于语义分割(recursive) 2:基于固定长度分割(character)

        # 检索配置
        TOP_K: ClassVar[int] = 3                                       # 返回最相似的K个结果
        SEARCH_TYPE: ClassVar[str] = "similarity_score_threshold"      # 检索类型算法  1:similarity 2:mmr 3:similarity_score_threshold
        SEARCH_SCORE_THRESHOLD: ClassVar[float] = 0.7                  # 需要根据实际效果调整

        # 新增混合检索配置
        RETRIEVER_STRATEGY: ClassVar[str] = "hybrid"                   # 稀疏检索(BM25), 稠密检索(hybrid)
        VECTOR_WEIGHT: ClassVar[float] = 0.6                           # VECTOR_WEIGHT + BM25_WEIGHT 权重之和建议为1
        BM25_WEIGHT: ClassVar[float] = 0.4                             # VECTOR_WEIGHT + BM25_WEIGHT 权重之和建议为1
        BM25_K1: ClassVar[float] = 1.5                                 # 控制词频饱和度, 值越大对高频词惩罚越强, 典型值1.2-2.0
        BM25_B: ClassVar[float] = 0.75                                 # 控制文档长度归一化, 0=禁用归一化, 1=完全归一化, 典型值0.5-0.8

        # 缓存配置
        ENABLE_CACHE: ClassVar[bool] = True                            # 建议：分类型启用缓存
        CACHE_TYPE: ClassVar[str] = "sqlite"                           # 合理选项
        CACHE_PATH: ClassVar[str] = ".rag_cache.db"                    # 建议：放在专用目录
        CACHE_MAX_SIZE: ClassVar[int] = 1000                           # 最大缓存条目数
        CACHE_TTL: ClassVar[Optional[int]] = 3600                      # 缓存过期时间(秒)，None表示永不过期
        CACHE_RETRY_INTERVAL: ClassVar[int] = 60                       # 初始化失败后重试间隔(秒)

        # FAISS专用配置
        FAISS_INDEX_TYPE: ClassVar[str] = "IVF"                        # 建议：IVF更高效 [Flat, IVF]
        FAISS_INDEX_PARAMS: ClassVar[dict] = {}                        # 需要根据数据量调整

        # 系统行为
        ALLOW_DANGEROUS_DESERIALIZATION: ClassVar[bool] = True         # 安全风险，建议有防护

        @classmethod
        def validate(cls):
                """配置有效性检查"""
                assert cls.TOP_K > 0, "TOP_K 必须大于0"
                assert cls.CHUNK_OVERLAP < cls.CHUNK_SIZE, "CHUNK_OVERLAP必须小于CHUNK_SIZE"
                assert cls.RETRIEVER_STRATEGY in ["hybrid", "vector", "bm25"], "无效检索策略"
                assert 0 <= cls.BM25_WEIGHT <= 1, "BM25权重需在0-1之间"
                assert 1.0 <= cls.BM25_K1 <= 2.0, "BM25_K1参数应在1.0-2.0之间"
                assert 0.5 <= cls.BM25_B <= 1.0, "BM25_B参数应在0.5-1.0之间"
                assert cls.FAISS_INDEX_TYPE in ["Flat", "IVF"], "无效的FAISS索引类型"
                if cls.FAISS_INDEX_TYPE == "IVF":
                        cls.FAISS_INDEX_PARAMS = {"nlist": 1024}  # 典型值

