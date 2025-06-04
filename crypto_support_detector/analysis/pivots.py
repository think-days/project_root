"""analysis/pivots.py - 支点检测模块"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import argrelextrema

from ..config.settings import config


class PivotDetector:
    """支点检测器"""

    def __init__(self, window: Optional[int] = None):
        self.window = window or config.get('analysis.pivot_window', 5)

    def detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测支点"""
        df = df.copy()

        # 使用滚动窗口检测局部最小值
        low = df['low']
        ref = low.rolling(self.window * 2 + 1, center=True).min()

        # 支点条件
        is_pivot = (
                (low == ref) &
                (low < low.shift(1)) &
                (low < low.shift(-1))
        )

        df['pivot'] = is_pivot

        # 增强检测：使用scipy查找更准确的局部极值
        if len(df) > self.window * 2:
            minima = argrelextrema(
                low.values, np.less, order=self.window
            )[0]

            enhanced_pivots = pd.Series(False, index=df.index)
            enhanced_pivots.iloc[minima] = True

            df['enhanced_pivot'] = enhanced_pivots
            df['final_pivot'] = df['pivot'] | df['enhanced_pivot']
        else:
            df['final_pivot'] = df['pivot']

        return df

    def get_pivot_points(self, df: pd.DataFrame) -> pd.Series:
        """获取支点价格"""
        df = self.detect_pivots(df)
        return df.loc[df['final_pivot'], 'low']

    def validate_pivots(self, df: pd.DataFrame, pivots: pd.Series) -> pd.Series:
        """验证支点的有效性"""
        validated = []

        for idx, pivot_price in pivots.items():
            # 检查支点前后的价格行为
            loc = df.index.get_loc(idx)

            if loc < 10 or loc > len(df) - 10:
                continue

            # 支点前的下降趋势
            pre_trend = df['low'].iloc[loc - 10:loc].is_monotonic_decreasing

            # 支点后的上升趋势
            post_trend = df['low'].iloc[loc:loc + 10].is_monotonic_increasing

            # 成交量确认
            vol_spike = df['volume'].iloc[loc] > df['volume'].iloc[loc - 10:loc].mean()

            if (pre_trend or post_trend) and vol_spike:
                validated.append((idx, pivot_price))

        if validated:
            return pd.Series(
                [v[1] for v in validated],
                index=[v[0] for v in validated]
            )
        return pd.Series(dtype=float)
