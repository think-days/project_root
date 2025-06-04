"""data/fetcher.py - 数据获取模块"""
import asyncio
import aiohttp
from typing import List, Dict, Optional, Callable
import pandas as pd
from datetime import datetime, timedelta
import logging

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt_async = None
    logging.warning("ccxt not installed, using HTTP API only")

from ..config.settings import config
from .models import MarketData
from .cache import DataCache

logger = logging.getLogger(__name__)

class SymbolFilter:
    """交易对过滤器"""

    def __init__(self):
        self.filters = config.get('data.symbol_filters', {})
        self.stablecoins = {
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX',
            'HUSD', 'GUSD', 'USDP', 'USDD', 'FRAX', 'LUSD'
        }

    async def filter_symbols(self, exchange, symbols: List[str]) -> List[str]:
        """过滤交易对"""
        filtered = []

        # 获取交易所信息
        markets = exchange.markets
        tickers = await exchange.fetch_tickers()

        for symbol in symbols:
            try:
                # 基础检查
                if symbol not in markets:
                    continue

                market = markets[symbol]
                ticker = tickers.get(symbol, {})

                # 1. 检查计价货币
                if market['quote'] not in self.filters.get('quote_currencies', ['USDT']):
                    continue

                # 2. 排除稳定币对
                if self.filters.get('exclude_stablecoin_pairs', True):
                    if (market['base'] in self.stablecoins and
                        market['quote'] in self.stablecoins):
                        continue

                # 3. 检查成交量
                volume_usdt = ticker.get('quoteVolume', 0)
                if volume_usdt < self.filters.get('min_volume_usdt', 1000000):
                    continue

                # 4. 检查上线时间（通过API获取）
                if not await self._check_symbol_age(exchange, symbol):
                    continue

                # 5. 检查标签（观察名单等）
                if self._has_excluded_tags(market):
                    continue

                filtered.append(symbol)

            except Exception as e:
                logger.debug(f"Error filtering {symbol}: {e}")
                continue

        # 限制最大数量
        max_symbols = self.filters.get('max_symbols', 200)
        if len(filtered) > max_symbols:
            # 按成交量排序，取前N个
            filtered = sorted(
                filtered,
                key=lambda s: tickers.get(s, {}).get('quoteVolume', 0),
                reverse=True
            )[:max_symbols]

        logger.info(f"Filtered {len(symbols)} symbols to {len(filtered)}")
        return filtered

    async def _check_symbol_age(self, exchange, symbol: str) -> bool:
        """检查交易对上线时间"""
        try:
            # 尝试获取历史数据判断上线时间
            min_age_days = self.filters.get('min_age_days', 30)

            # 获取最早的K线数据
            since = exchange.milliseconds() - (min_age_days + 1) * 24 * 60 * 60 * 1000
            ohlcv = await exchange.fetch_ohlcv(
                symbol, '1d', since=since, limit=min_age_days + 1
            )

            # 如果有足够的历史数据，说明不是新币
            return len(ohlcv) >= min_age_days

        except Exception:
            # 出错时保守处理，不包含该交易对
            return False

    def _has_excluded_tags(self, market: Dict) -> bool:
        """检查是否有排除的标签"""
        excluded_tags = self.filters.get('exclude_tags', ['monitoring', 'assessment'])
        market_info = market.get('info', {})

        # Binance特定的标签检查
        tags = market_info.get('tags', [])
        permissions = market_info.get('permissions', [])

        for tag in excluded_tags:
            if tag in tags or tag in permissions:
                return True

        return False

class DataFetcher:
    """异步数据获取器"""

    def __init__(self, exchange: str = "binance"):
        self.exchange_name = exchange
        self.exchange = None
        self.cache = DataCache()
        self.symbol_filter = SymbolFilter()
        self.rate_limiter = RateLimiter(config.get('data.rate_limit', 1200))
        self._session = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()

    async def initialize(self):
        """初始化交换所连接"""
        if self.exchange_name == "binance":
            self.exchange = ccxt_async.binance({
                'enableRateLimit': True,
                'rateLimit': 50,
                'options': {
                    'defaultType': 'spot',  # 现货交易
                }
            })
        await self.exchange.load_markets()
        self._session = aiohttp.ClientSession()

    async def close(self):
        """关闭连接"""
        if self.exchange:
            await self.exchange.close()
        if self._session:
            await self._session.close()

    async def fetch_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """获取过滤后的交易对列表"""
        try:
            # 获取所有活跃的交易对
            all_symbols = []
            for symbol, market in self.exchange.markets.items():
                if (market['quote'] == quote_currency and
                    market['active'] and
                    market['spot']):  # 只要现货
                    all_symbols.append(symbol)

            # 应用过滤器
            filtered_symbols = await self.symbol_filter.filter_symbols(
                self.exchange, all_symbols
            )

            return filtered_symbols

        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str,
                         limit: Optional[int] = None) -> Optional[MarketData]:
        """获取单个交易对的OHLCV数据"""
        cache_key = f"{symbol}:{timeframe}:{limit}"

        # 检查缓存
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        try:
            await self.rate_limiter.acquire()

            if limit is None:
                limit = config.get(f'analysis.timeframes.{timeframe}.limit', 100)

            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 添加技术指标
            df = self._add_technical_indicators(df)

            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                df=df
            )

            # 缓存数据
            await self.cache.set(cache_key, market_data,
                               ttl=config.get('data.cache_ttl', 300))

            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 计算ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # 计算成交量移动平均
        df['volume_ma'] = df['volume'].rolling(20).mean()

        # 计算价格变化率
        df['returns'] = df['close'].pct_change()

        return df

    async def fetch_multiple(self, symbols: List[str], timeframe: str,
                           callback: Optional[Callable] = None) -> Dict[str, MarketData]:
        """批量获取多个交易对数据"""
        # 限制并发数
        max_concurrent = config.get('monitoring.max_concurrent', 50)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return await self.fetch_ohlcv(symbol, timeframe)

        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]

        results = {}
        for symbol, result in zip(symbols, await asyncio.gather(*tasks)):
            if result:
                results[symbol] = result
                if callback:
                    callback(symbol, result)

        return results

class RateLimiter:
    """速率限制器"""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.semaphore = asyncio.Semaphore(max_per_minute // 60)  # 每秒的请求数
        self.request_times = []

    async def acquire(self):
        """获取许可"""
        await self.semaphore.acquire()

        # 清理超过1分钟的记录
        now = datetime.now()
        self.request_times = [
            t for t in self.request_times
            if now - t < timedelta(minutes=1)
        ]

        # 如果1分钟内请求过多，等待
        if len(self.request_times) >= self.max_per_minute:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_times.append(now)
        self.semaphore.release()
