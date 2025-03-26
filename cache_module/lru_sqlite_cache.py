#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/26 10:39
# @Author  : fengzhengxiong
# @File    : lru_sqlite_cache.py


import sqlite3
from typing import Optional, Dict, Any
from langchain.cache import SQLiteCache


class LRUSQLiteCache(SQLiteCache):
    """支持LRU淘汰策略的增强版SQLite缓存"""

    def __init__(self, database_path: str, max_size: int = 1000):
        """
        初始化LRU SQLite缓存

        Args:
            database_path: SQLite数据库文件路径
            max_size: 最大缓存条目数 (达到上限时触发LRU淘汰)
        """
        # 先调用父类初始化
        super().__init__(database_path)
        self.max_size = max(max_size, 1)  # 确保至少保留1个条目

        # 直接创建数据库连接
        self.conn = sqlite3.connect(database_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # 初始化数据库结构
        self._setup_database()

    def _setup_database(self) -> None:
        """初始化数据库表结构和索引"""
        try:
            with self.conn:
                # 创建支持LRU的主表
                self.conn.execute("""
                                CREATE TABLE IF NOT EXISTS lru_cache (
                                    id INTEGER PRIMARY KEY,
                                    prompt TEXT NOT NULL,
                                    llm_string TEXT NOT NULL,
                                    response TEXT NOT NULL,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    UNIQUE(prompt, llm_string)  -- 复合唯一键
                                )
                                """)

                # 创建加速查询的索引
                self.conn.execute("""
                                CREATE INDEX IF NOT EXISTS idx_cache_lookup 
                                ON lru_cache(prompt, llm_string)
                                """)

                self.conn.execute("""
                                CREATE INDEX IF NOT EXISTS idx_cache_access 
                                ON lru_cache(last_accessed)
                                """)

                # 创建自动淘汰的触发器
                self.conn.execute(f"""
                                CREATE TRIGGER IF NOT EXISTS trigger_lru_eviction
                                AFTER INSERT ON lru_cache
                                WHEN (SELECT COUNT(*) FROM lru_cache) > {self.max_size}
                                BEGIN
                                    DELETE FROM lru_cache 
                                    WHERE id IN (
                                        SELECT id FROM lru_cache 
                                        ORDER BY last_accessed ASC 
                                        LIMIT (SELECT COUNT(*) FROM lru_cache) - {self.max_size}
                                    );
                                END;
                                """)
                self.conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"初始化LRU缓存数据库失败: {str(e)}")

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """
        查找缓存项并更新访问时间

        Args:
            prompt: 用户输入的提示词
            llm_string: 语言模型配置标识

        Returns:
            缓存的响应内容，未命中时返回None
        """
        try:
            with self.conn:
                cursor = self.conn.execute("""
                            SELECT response FROM lru_cache 
                            WHERE prompt = ? AND llm_string = ?
                            """, (str(prompt), str(llm_string)))

                if result := cursor.fetchone():
                    # 更新访问时间
                    self.conn.execute("""
                                UPDATE lru_cache 
                                SET last_accessed = CURRENT_TIMESTAMP 
                                WHERE prompt = ? AND llm_string = ?
                                """, (str(prompt), str(llm_string)))
                    self.conn.commit()
                    return result[0]
                return None
        except sqlite3.Error as e:
            self._handle_error("查找缓存失败", e)
            return None


    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        """
        更新或插入缓存项

        Args:
            prompt: 用户输入的提示词
            llm_string: 语言模型配置标识
            return_val: 要缓存的语言模型响应
        """
        try:
            with self.conn:
                self.conn.execute("""
                            INSERT OR REPLACE INTO lru_cache 
                            (prompt, llm_string, response, last_accessed)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                            """, (str(prompt), str(llm_string), str(return_val)))
                self.conn.commit()
        except sqlite3.Error as e:
            self._handle_error("更新缓存失败", e)


    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "type": "lru_sqlite",
            "max_size": self.max_size,
            "status": "active"
        }

        try:
            with self.conn:
                stats.update({
                    "current_size": self.conn.execute(
                        "SELECT COUNT(*) FROM lru_cache").fetchone()[0],
                    "oldest_entry": self.conn.execute(
                        "SELECT MIN(last_accessed) FROM lru_cache").fetchone()[0],
                    "newest_entry": self.conn.execute(
                        "SELECT MAX(last_accessed) FROM lru_cache").fetchone()[0],
                    "hit_rate": self._calculate_hit_rate(self.conn)
                })
        except sqlite3.Error as e:
            stats["status"] = f"error: {str(e)}"

        return stats

    def clear(self) -> None:
        """清空整个缓存"""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM lru_cache")
                self.conn.commit()
        except sqlite3.Error as e:
            self._handle_error("清空缓存失败", e)


    def _calculate_hit_rate(self, conn: sqlite3.Connection) -> float:
        """计算缓存命中率(需要额外统计表)"""
        # 实际实现需要创建统计表记录查询次数和命中次数
        return 0.0  # 示例返回值

    def _handle_error(self, operation: str, error: sqlite3.Error) -> None:
        """统一错误处理"""
        error_msg = f"{operation}: {str(error)}"
        # 这里可以添加更详细的错误日志记录
        print(f"⚠️ 缓存错误: {error_msg}")