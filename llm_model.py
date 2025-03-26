#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 13:44
# @Author  : fengzhengxiong
# @File    : llm_model.py

import logging
from langchain_community.llms import Ollama

from config_module import Config
from log_module import RAGLogger

class LLMModel:
    def __init__(self):
        self.logger = RAGLogger().get_logger()
        # self.logger = logging.getLogger(__name__)
        self.model = self._init_model()
        # 优化建议：可以增加模型健康检查

    def _init_model(self) -> Ollama:
        """初始化Ollama模型"""
        try:
            return Ollama(
                model=Config.LLM_MODEL,                   # 指定Ollama模型名称
                temperature=Config.LLM_TEMPERATURE,       # 控制输出随机性, 0(确定)-1(随机), 知识问答建议0.3-0.7
                num_ctx=Config.LLM_MAX_TOKENS,            # 上下文token限制, 取决于模型, 不应超过模型最大值
                top_k=40,                                 # 候选词筛选数量, 10-100, 值越大结果越多样
                top_p=0.9,                                # 概率累积阈值, 0.5-1.0, 与temperature配合调节
                repeat_penalty=1.1                        # 重复惩罚系数, 1.0-1.5, 抑制重复内容生成
            )
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise RuntimeError("无法加载语言模型，请检查Ollama服务是否运行")
            # 优化建议：可以增加备用模型机制

    def generate(self, prompt: str) -> str:
        """安全生成文本"""
        try:
            # 同步调用：invoke()是阻塞式调用
            return self.model.invoke(prompt)
            # 优化建议：可以增加流式输出支持
        except Exception as e:
            self.logger.error(f"生成失败: {str(e)}")
            return "抱歉，生成回答时出错"
            # 优化建议：可以区分不同类型错误