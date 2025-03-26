#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:39
# @Author  : fengzhengxiong
# @File    : lru_memory_cache.py

import threading
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple
from langchain.schema import Generation


class LRUInMemoryCache:
    """
    线程安全的高性能内存LRU缓存实现
    支持缓存统计、TTL过期和动态大小调整

    特性：
    - 严格的LRU淘汰策略
    - 线程安全操作
    - 缓存命中率统计
    - 可选的TTL过期
    - 动态调整缓存大小
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        初始化内存缓存

        Args:
            max_size: 最大缓存条目数 (默认1000)
            ttl: 缓存过期时间(秒)，None表示永不过期
        """
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.max_size = max(max_size, 1)
        self.ttl = ttl
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }

    def lookup(self, prompt: str, llm_string: str) -> Optional[Generation]:
        """
        查找缓存项并更新访问时间

        Args:
            prompt: 用户输入的提示词
            llm_string: 语言模型配置标识

        Returns:
            Generation对象(包含缓存的响应)，未命中时返回None
        """
        key = self._generate_key(prompt, llm_string)

        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None

            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                self._stats['expired'] += 1
                self._stats['misses'] += 1
                return None

            # 更新为最近使用
            self._cache.move_to_end(key)
            entry['last_accessed'] = datetime.now()
            self._stats['hits'] += 1

            return Generation(text=entry['response'])

    def update(self, prompt: str, llm_string: str, return_val: Generation) -> None:
        """
        更新或插入缓存项

        Args:
            prompt: 用户输入的提示词
            llm_string: 语言模型配置标识
            return_val: 要缓存的Generation对象
        """
        key = self._generate_key(prompt, llm_string)
        now = datetime.now()

        with self._lock:
            # 如果已存在则先删除 (确保LRU顺序正确)
            if key in self._cache:
                del self._cache[key]

            # 执行LRU淘汰
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._stats['evictions'] += 1

            # 添加新条目
            self._cache[key] = {
                'response': return_val.text,
                'created_at': now,
                'last_accessed': now
            }

    def clear(self) -> None:
        """清空整个缓存"""
        with self._lock:
            self._cache.clear()
            self._reset_stats()

    def resize(self, new_size: int) -> None:
        """
        动态调整缓存大小

        Args:
            new_size: 新的最大缓存条目数
        """
        with self._lock:
            self.max_size = max(new_size, 1)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self._stats['evictions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含以下信息的字典:
            - current_size: 当前缓存条目数
            - max_size: 最大容量
            - hits: 缓存命中次数
            - misses: 缓存未命中次数
            - hit_rate: 命中率(0-1)
            - evictions: LRU淘汰次数
            - expired: 过期条目数
            - oldest: 最旧条目的访问时间
            - newest: 最新条目的访问时间
        """
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0

            oldest = min(
                (entry['last_accessed'] for entry in self._cache.values()),
                default=None
            )
            newest = max(
                (entry['last_accessed'] for entry in self._cache.values()),
                default=None
            )

            return {
                'type': 'lru_memory',
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': round(hit_rate, 4),
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired'],
                'oldest_entry': oldest.isoformat() if oldest else None,
                'newest_entry': newest.isoformat() if newest else None,
                'ttl': self.ttl
            }

    def _generate_key(self, prompt: str, llm_string: str) -> Tuple[str, str]:
        """生成统一的缓存键"""
        return (prompt, llm_string)

    def _is_expired(self, entry: Dict) -> bool:
        """检查缓存条目是否过期"""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - entry['last_accessed']).total_seconds()
        return elapsed > self.ttl

    def _reset_stats(self) -> None:
        """重置统计计数器"""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }