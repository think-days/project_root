#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py - 主程序入口"""
import asyncio
import argparse
import logging
import signal
import sys
from typing import List, Optional

from crypto_support_detector.config.settings import config
from crypto_support_detector.monitoring.scanner import RealTimeScanner
from crypto_support_detector.utils.logger import setup_logging

# 设置日志
logger = setup_logging()

class Application:
    """主应用程序"""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.scanner = RealTimeScanner(symbols)
        self.running = False

    async def start(self):
        """启动应用"""
        logger.info("Starting Crypto Support Detector...")

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # 启动扫描器
        await self.scanner.start()

        # 主循环
        while self.running:
            await asyncio.sleep(1)

            # 定期输出统计信息
            if int(asyncio.get_event_loop().time()) % 60 == 0:
                self._print_stats()

    async def stop(self):
        """停止应用"""
        logger.info("Stopping application...")
        self.running = False
        await self.scanner.stop()

    def _signal_handler(self, sig, frame):
        """信号处理"""
        logger.info(f"Received signal {sig}")
        self.running = False

    def _print_stats(self):
        """打印统计信息"""
        results = self.scanner.get_results()
        total_symbols = len(results)
        total_supports = sum(
            len(tf_data.get('levels', []))
            for symbol_data in results.values()
            for tf_data in symbol_data.values()
        )

        logger.info(f"Stats: {total_symbols} symbols, {total_supports} support levels")

        # 显示最强支撑位
        top_supports = self.scanner.get_top_supports(5)
        for i, item in enumerate(top_supports, 1):
            level = item['level']
            logger.info(
                f"Top {i}: {item['symbol']} {item['timeframe']} - "
                f"Level: {level.level:.6f}, Score: {level.score:.3f}, "
                f"Category: {level.category}"
            )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Real-time cryptocurrency support level detector"
    )

    parser.add_argument(
        '-s', '--symbols',
        nargs='+',
        help='Symbols to scan (e.g., BTC/USDT ETH/USDT)'
    )

    parser.add_argument(
        '-c', '--config',
        help='Configuration file path'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 加载配置
    if args.config:
        config.load_from_file(args.config)

    # 创建并启动应用
    app = Application(args.symbols)

    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
