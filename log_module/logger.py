#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 11:21
# @Author  : fengzhengxiong
# @File    : logger.py


import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class RAGLogger:
    """
    RAG系统专用日志记录器
    支持：
    - 多级别日志记录
    - 日志文件轮转
    - 控制台和文件双重输出
    """

    _instance = None  # 单例模式

    def __new__(cls, name: str = "RAG", log_dir: str = "logs",
                max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(name, log_dir, max_bytes, backup_count)
        return cls._instance

    def _initialize(self, name, log_dir, max_bytes, backup_count):
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志目录路径
            max_bytes: 单个日志文件最大大小(字节)
            backup_count: 保留的备份文件数量
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 确保日志目录存在
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 主日志文件路径
        self.main_log_path = self.log_dir / "rag.log"

        # 配置日志格式
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 初始化处理器
        self._setup_handlers(max_bytes, backup_count)

    def _setup_handlers(self, max_bytes: int, backup_count: int) -> None:
        """配置日志处理器"""
        # 清空现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 主日志文件处理器(带轮转)
        file_handler = logging.handlers.RotatingFileHandler(
            self.main_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)

        # 添加所有处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        """记录信息级别日志"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """记录警告级别日志"""
        self.logger.warning(message)

    def error(self, message: str, exc_info: Optional[bool] = False) -> None:
        """
        记录错误级别日志

        Args:
            message: 错误消息
            exc_info: 是否包含异常信息
        """
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: Optional[bool] = False) -> None:
        """
        记录严重错误级别日志

        Args:
            message: 错误消息
            exc_info: 是否包含异常信息
        """
        self.logger.critical(message, exc_info=exc_info)

    def get_logger(self) -> logging.Logger:
        """获取底层logging.Logger对象"""
        return self.logger

    def change_log_level(self, level: int) -> None:
        """
        动态修改日志级别

        Args:
            level: 日志级别 (logging.DEBUG/INFO/WARNING/ERROR/CRITICAL)
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)