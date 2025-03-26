#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:38
# @Author  : fengzhengxiong
# @File    : cache_manager.py

import langchain
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

from . import LRUSQLiteCache, LRUInMemoryCache
from config_module import Config
from log_module import RAGLogger

class CacheManager:
    """增强版缓存管理单例，支持LRU策略和多维度监控"""

    _instance: Optional['CacheManager'] = None
    logger = RAGLogger().get_logger()

    def __new__(cls) -> 'CacheManager':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化实例状态"""
        self._cache_initialized: bool = False
        self._init_attempts: int = 0
        self._last_error: Optional[str] = None
        self._init_cache()

    def _init_cache(self) -> None:
        """初始化缓存系统"""
        if not Config.ENABLE_CACHE:
            langchain.llm_cache = None
            self.logger.info("缓存功能已禁用")
            return

        try:
            self._init_attempts += 1

            if Config.CACHE_TYPE == "sqlite":
                self._setup_sqlite_cache()
            else:
                self._setup_memory_cache()

            self._cache_initialized = True
            self.logger.info(
                f"成功初始化 {Config.CACHE_TYPE} 缓存，"
                f"最大容量: {Config.CACHE_MAX_SIZE}, "
                f"TTL: {getattr(Config, 'CACHE_TTL', '无')}"
            )

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"缓存初始化失败 (尝试次数: {self._init_attempts}): {self._last_error}")
            langchain.llm_cache = None
            self._cache_initialized = False

    def _setup_sqlite_cache(self) -> None:
        """配置SQLite缓存"""
        cache_path = Path(Config.CACHE_PATH)

        # 确保缓存目录存在
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        langchain.llm_cache = LRUSQLiteCache(
            database_path=str(cache_path),
            max_size=Config.CACHE_MAX_SIZE
        )

    def _setup_memory_cache(self) -> None:
        """配置内存缓存"""
        langchain.llm_cache = LRUInMemoryCache(
            max_size=Config.CACHE_MAX_SIZE,
            ttl=getattr(Config, 'CACHE_TTL', None)
        )

    @classmethod
    def clear_cache(cls) -> Dict[str, Any]:
        """
        清空缓存并返回操作结果

        Returns:
            包含操作状态的字典:
            - success: 是否成功
            - message: 详细信息
            - cache_type: 缓存类型
        """
        result = {
            'success': False,
            'message': '',
            'cache_type': Config.CACHE_TYPE
        }

        try:
            if Config.CACHE_TYPE == "sqlite":
                cache_path = Path(Config.CACHE_PATH)
                if cache_path.exists():
                    cache_path.unlink()
                    result['success'] = True
                    result['message'] = "SQLite缓存已清空"
                    cls.logger.info(result['message'])
                else:
                    result['message'] = "SQLite缓存文件不存在"
                    cls.logger.warning(result['message'])

            elif hasattr(langchain.llm_cache, 'clear'):
                langchain.llm_cache.clear()
                result['success'] = True
                result['message'] = "内存缓存已清空"
                cls.logger.info(result['message'])

            else:
                result['message'] = "未知缓存类型或缓存未初始化"
                cls.logger.error(result['message'])

        except Exception as e:
            result['message'] = f"清空缓存失败: {str(e)}"
            cls.logger.error(result['message'])

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取详细的缓存统计信息

        Returns:
            包含以下信息的字典:
            - status: 缓存状态
            - type: 缓存类型
            - size: 当前大小/最大容量
            - hit_rate: 命中率(如可用)
            - initialized: 是否已初始化
            - last_error: 最后错误信息(如有)
            - init_attempts: 初始化尝试次数
        """
        base_stats = {
            'status': 'active' if self._cache_initialized else 'inactive',
            'type': Config.CACHE_TYPE,
            'initialized': self._cache_initialized,
            'last_error': self._last_error,
            'init_attempts': self._init_attempts
        }

        if hasattr(langchain.llm_cache, 'get_stats'):
            try:
                cache_stats = langchain.llm_cache.get_stats()
                base_stats.update({
                    'size': f"{cache_stats.get('current_size', '?')}/{Config.CACHE_MAX_SIZE}",
                    'hit_rate': cache_stats.get('hit_rate', 'N/A'),
                    'details': cache_stats
                })
            except Exception as e:
                base_stats['status'] = f"error: {str(e)}"
                self.logger.error(f"获取缓存统计失败: {str(e)}")
        else:
            base_stats['status'] = 'not_available'

        return base_stats

    def health_check(self) -> Dict[str, Any]:
        """
        执行缓存健康检查

        Returns:
            包含健康状态和详细信息的字典
        """
        health = {
            'healthy': False,
            'message': '',
            'cache_type': Config.CACHE_TYPE
        }

        if not Config.ENABLE_CACHE:
            health.update({
                'healthy': True,
                'message': '缓存功能已禁用'
            })
            return health

        try:
            if Config.CACHE_TYPE == "sqlite":
                # 测试SQLite数据库连接
                with sqlite3.connect(Config.CACHE_PATH) as conn:
                    conn.execute("SELECT 1")
                health.update({
                    'healthy': True,
                    'message': 'SQLite缓存连接正常'
                })

            elif hasattr(langchain.llm_cache, 'lookup'):
                # 测试内存缓存基本功能
                test_key = ("health_check", "test")
                langchain.llm_cache.update(*test_key, "test_value")
                if langchain.llm_cache.lookup(*test_key) == "test_value":
                    health.update({
                        'healthy': True,
                        'message': '内存缓存功能正常'
                    })
                else:
                    health['message'] = '内存缓存功能异常'

            else:
                health['message'] = '未知缓存类型'

        except Exception as e:
            health['message'] = f"健康检查失败: {str(e)}"

        return health