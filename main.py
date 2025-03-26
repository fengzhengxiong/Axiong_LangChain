#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/22 11:59
# @Author  : fengzhengxiong
# @File    : main.py

"""
      ┏┛ ┻━━━━━┛ ┻┓
      ┃　　　　　　 ┃
      ┃　　　━　　　┃
      ┃　┳┛　  ┗┳　┃
      ┃　　　　　　 ┃
      ┃　　　┻　　　┃
      ┃　　　　　　 ┃
      ┗━┓　　　┏━━━┛
        ┃　　　┃   神兽保佑
        ┃　　　┃   代码正常
        ┃　　　┗━━━━━━━━━┓
        ┃               ┣┓
        ┃　　　　         ┏┛
        ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
          ┃ ┫ ┫   ┃ ┫ ┫
          ┗━┻━┛   ┗━┻━┛

"""

import time
from log_module import RAGLogger
from rag_system import RAGSystem
from cache_module import CacheManager

def print_banner() -> None:
    """ 显示系统横幅 """
    banner = """
    ***************************************
    * RAG系统（混合检索+缓存优化版）  *
    * 版本: 0.1                  *
    * 支持命令:                  *
    *   q - 退出               *
    *   c - 清空缓存           *
    *   t - 检索测试          *
    *   s - 系统状态         *
    ***************************************
    """
    print(banner)


def execute_query(rag: RAGSystem, query: str, logger: RAGLogger) -> str:
    """
    执行查询并返回结果
    包含性能监控和错误处理
    """
    try:
        start_time = time.perf_counter()
        result = rag.ask(query)
        elapsed = time.perf_counter() - start_time

        logger.info(f"查询完成 - 耗时: {elapsed:.2f}s - 问题: {query[:50]}...")
        return result

    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}", exc_info=True)
        return "系统处理您的请求时出错，请稍后再试"


def handle_command(rag: RAGSystem, cmd: str, logger: RAGLogger) -> bool:
    """
    处理用户命令
    返回是否继续运行
    """
    try:
        if cmd.lower() == 'q':  # 退出
            return False

        elif cmd.lower() == 'c':  # 清空缓存
            result = CacheManager.clear_cache()
            print(f"\n缓存清理结果: {result['message']}")
            logger.info(f"用户手动清空缓存 - 结果: {result}")

        elif cmd.lower() == 't':  # 测试检索
            query = input("\n输入测试查询: ").strip()
            if query:
                rag.test_retrievers(query)
            else:
                print("测试查询不能为空")

        elif cmd.lower() == 's':  # 系统状态
            show_system_status(rag, logger)

        elif cmd:  # 正常查询
            print("\n思考中...")
            answer = execute_query(rag, cmd, logger)
            print(f"\nAI: {answer}")

    except KeyboardInterrupt:
        print("\n操作已取消")
        return True
    except Exception as e:
        logger.error(f"命令处理错误: {str(e)}", exc_info=True)
        print(f"\n系统错误: {str(e)}")

    return True


def show_system_status(rag: RAGSystem, logger: RAGLogger) -> None:
    """显示系统状态信息"""
    cache_stats = CacheManager().get_cache_stats()
    print("\n=== 系统状态 ===")
    print(f"缓存状态: {cache_stats.get('status', '未知')}")
    print(f"缓存类型: {cache_stats.get('type', '未知')}")
    print(f"缓存使用: {cache_stats.get('size', '?')}")
    print(f"命中率: {cache_stats.get('hit_rate', 'N/A')}")

    if hasattr(rag, 'get_system_metrics'):
        metrics = rag.get_system_metrics()
        print("\n--- RAG指标 ---")
        print(f"文档数量: {metrics.get('document_count', '未知')}")
        print(f"平均检索时间: {metrics.get('avg_retrieval_time', '未知')}ms")


def main() -> None:
    """主程序入口"""
    # 初始化日志系统
    logger = RAGLogger(name="RAGSystem", log_dir="logs")

    try:
        logger.info("系统启动初始化开始")
        start_time = time.perf_counter()

        # 初始化缓存
        cache_manager = CacheManager()

        # 检查缓存健康状态
        if hasattr(cache_manager, 'health_check'):
            cache_health = cache_manager.health_check()
            logger.info(f"缓存健康状态: {cache_health}")

        # 初始化RAG系统
        rag = RAGSystem()
        init_time = time.perf_counter() - start_time
        logger.info(f"系统初始化完成，耗时: {init_time:.2f}秒")

        print_banner()

        # 主循环
        while True:
            try:
                cmd = input("\n请输入问题或命令(q退出): ").strip()
                if not handle_command(rag, cmd, logger):
                    break

            except KeyboardInterrupt:
                print("\n接收到中断信号，准备退出...")
                break
            except Exception as e:
                logger.critical(f"主循环异常: {str(e)}", exc_info=True)
                print("\n系统发生严重错误，请查看日志")
                break

    except Exception as e:
        logger.critical(f"系统启动失败: {str(e)}", exc_info=True)
        print(f"\n系统启动失败: {str(e)}")
    finally:
        logger.info("系统安全关闭")
        print("\n系统已安全退出")


if __name__ == "__main__":
    main()