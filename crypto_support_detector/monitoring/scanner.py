"""monitoring/scanner.py - 实时扫描器"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from ..config.settings import config
from ..data.fetcher import DataFetcher
from ..data.models import MarketData, SupportLevel
from ..analysis.clustering import SupportLevelClustering
from ..analysis.scoring import SupportScoring
from .alerts import AlertManager

logger = logging.getLogger(__name__)



class RealTimeScanner:
    """实时支撑位扫描器"""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or []
        self.fetcher = None
        self.clustering = SupportLevelClustering()
        self.scoring = SupportScoring()
        self.alert_manager = AlertManager()

        self.scan_interval = config.get('monitoring.scan_interval', 60)
        self.timeframes = list(config.get('analysis.timeframes', {}).keys())

        self._running = False
        self._tasks = []
        self._results = defaultdict(dict)

    async def start(self):
        """启动扫描器"""
        logger.info("Starting real-time scanner...")

        self.fetcher = DataFetcher()
        await self.fetcher.initialize()

        # 如果没有指定交易对，获取所有活跃交易对
        if not self.symbols:
            self.symbols = await self.fetcher.fetch_symbols("USDT")
            logger.info(f"Loaded {len(self.symbols)} symbols")

        self._running = True

        # 为每个时间框架创建扫描任务
        for timeframe in self.timeframes:
            task = asyncio.create_task(self._scan_loop(timeframe))
            self._tasks.append(task)

        # 启动告警管理器
        await self.alert_manager.start()

    async def stop(self):
        """停止扫描器"""
        logger.info("Stopping scanner...")
        self._running = False

        # 取消所有任务
        for task in self._tasks:
            task.cancel()

        # 等待任务完成
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # 关闭连接
        if self.fetcher:
            await self.fetcher.close()

        await self.alert_manager.stop()

    async def _scan_loop(self, timeframe: str):
        """扫描循环"""
        while self._running:
            try:
                start_time = datetime.now()

                # 批量获取数据
                logger.info(f"Scanning {len(self.symbols)} symbols on {timeframe}")

                # 分批处理，避免过载
                batch_size = 50
                for i in range(0, len(self.symbols), batch_size):
                    batch = self.symbols[i:i + batch_size]

                    market_data_dict = await self.fetcher.fetch_multiple(
                        batch, timeframe
                    )

                    # 分析每个交易对
                    for symbol, market_data in market_data_dict.items():
                        support_levels = await self._analyze_symbol(
                            symbol, market_data
                        )

                        # 存储结果
                        self._results[symbol][timeframe] = {
                            'levels': support_levels,
                            'timestamp': datetime.now()
                        }

                        # 检查告警条件
                        await self._check_alerts(symbol, timeframe,
                                                 market_data, support_levels)

                # 记录扫描时间
                scan_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Scan completed for {timeframe} in {scan_time:.2f}s")

            except Exception as e:
                logger.error(f"Error in scan loop for {timeframe}: {e}")

            # 等待下次扫描
            await asyncio.sleep(self.scan_interval)

    async def _analyze_symbol(self, symbol: str,
                              market_data: MarketData) -> List[SupportLevel]:
        """分析单个交易对"""
        try:
            # 聚类分析
            clusters = self.clustering.cluster_levels(market_data.df)

            if not clusters:
                return []

            # 过滤并评分
            current_price = market_data.current_price
            allow_margin = config.get('analysis.allow_margin', 0.02)

            support_levels = []

            for cluster_id, cluster_data in clusters.items():
                level = cluster_data['level']

                # 过滤高于现价的支撑位
                if level > current_price * (1 + allow_margin):
                    continue

                # 创建支撑位对象
                support_level = self.scoring.create_support_level(
                    level, market_data.df, cluster_data
                )

                support_levels.append(support_level)

            # 按分数排序
            support_levels.sort(key=lambda x: x.score, reverse=True)

            return support_levels

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    async def _check_alerts(self, symbol: str, timeframe: str,
                            market_data: MarketData,
                            support_levels: List[SupportLevel]):
        """检查告警条件"""
        if not support_levels:
            return

        current_price = market_data.current_price

        for level in support_levels:
            # 检查是否接近强支撑位
            if level.category == "STRONG":
                distance_pct = (current_price - level.level) / level.level

                # 价格接近支撑位（2%以内）
                if 0 < distance_pct < 0.02:
                    await self.alert_manager.send_alert(
                        symbol=symbol,
                        timeframe=timeframe,
                        alert_type="APPROACHING_SUPPORT",
                        level=level,
                        current_price=current_price,
                        distance_pct=distance_pct
                    )

                # 价格刚突破支撑位
                elif -0.01 < distance_pct < 0:
                    await self.alert_manager.send_alert(
                        symbol=symbol,
                        timeframe=timeframe,
                        alert_type="SUPPORT_BREAK",
                        level=level,
                        current_price=current_price,
                        distance_pct=distance_pct
                    )

    def get_results(self, symbol: Optional[str] = None) -> Dict:
        """获取扫描结果"""
        if symbol:
            return self._results.get(symbol, {})
        return dict(self._results)

    def get_top_supports(self, n: int = 10) -> List[Dict]:
        """获取最强的支撑位"""
        all_supports = []

        for symbol, timeframes in self._results.items():
            for timeframe, data in timeframes.items():
                for level in data.get('levels', []):
                    all_supports.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'level': level,
                        'timestamp': data['timestamp']
                    })

        # 按分数排序
        all_supports.sort(key=lambda x: x['level'].score, reverse=True)

        return all_supports[:n]
