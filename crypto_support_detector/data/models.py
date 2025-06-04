"""data/models.py - 数据模型定义"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class OHLCV:
    """OHLCV数据模型"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCV':
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class SupportLevel:
    """支撑位数据模型"""
    level: float
    score: float
    confidence: float
    category: str
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 验证数据
        assert 0 <= self.score <= 1, "Score must be between 0 and 1"
        assert 0 <= self.confidence <= 1, "Confidence must be between 0 and 1"
        assert self.category in ["STRONG", "GOOD", "FAIR", "WEAK"]


@dataclass
class MarketData:
    """市场数据容器"""
    symbol: str
    timeframe: str
    df: pd.DataFrame
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def current_price(self) -> float:
        return float(self.df['close'].iloc[-1])

    @property
    def period_range(self) -> tuple:
        return self.df.index[0], self.df.index[-1]

    def get_atr(self, period: int = 14) -> float:
        """计算ATR"""
        df = self.df.copy()
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return float(df['tr'].tail(period).mean())
