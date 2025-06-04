"""analysis/scoring.py - 支撑位评分系统"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ..config.settings import config
from ..data.models import SupportLevel


class SupportScoring:
    """支撑位评分系统"""

    def __init__(self):
        self.weights_by_timeframe = config.get('scoring.weights', {})
        self.thresholds = config.get('scoring.thresholds', {})
        self.rebound_window = config.get('scoring.rebound_window', 10)
        self.volume_ma_window = config.get('scoring.volume_ma_window', 20)

    def calculate_components(self, level: float, df: pd.DataFrame,
                             cluster_data: Dict, timeframe: str) -> Dict[str, float]:
        """计算评分组件"""
        eps = config.get('analysis.base_eps_pct', 0.003)
        mask = (df['low'] >= level * (1 - eps)) & (df['low'] <= level * (1 + eps))

        # 获取时间框架特定的参数
        tf_config = config.get(f'analysis.timeframes.{timeframe}', {})
        min_touches = tf_config.get('min_touches', 2)

        components = {
            'touches': self._calculate_touches(mask, cluster_data, min_touches, timeframe),
            'rebound': self._calculate_rebound(level, df, mask, timeframe),
            'volume': self._calculate_volume(df, mask, timeframe),
            'recency': self._calculate_recency(df, mask, cluster_data, timeframe)
        }

        return components

    def _calculate_touches(self, mask: pd.Series, cluster_data: Dict,
                           min_touches: int, timeframe: str) -> float:
        """计算触及次数分数（考虑时间框架）"""
        # 结合mask计数和聚类数据
        mask_count = int(mask.sum())
        cluster_count = cluster_data.get('count', 0)
        touches = max(mask_count, cluster_count)

        # 时间框架调整
        if timeframe in ['1h', '4h']:
            # 短时间框架需要更多触及
            normalization_factor = 15
        elif timeframe in ['1d', '3d']:
            # 中等时间框架
            normalization_factor = 10
        else:  # 1w
            # 长时间框架触及次数自然较少
            normalization_factor = 5

        # 考虑最小触及次数
        if touches < min_touches:
            return 0.0

        return min(touches / normalization_factor, 1.0)

    def _calculate_rebound(self, level: float, df: pd.DataFrame,
                           mask: pd.Series, timeframe: str) -> float:
        """计算反弹强度（考虑时间框架）"""
        if not mask.any():
            return 0.0

        # 根据时间框架调整反弹窗口
        window_multipliers = {
            '1h': 0.5,  # 5个K线
            '4h': 0.7,  # 7个K线
            '1d': 1.0,  # 10个K线
            '3d': 1.2,  # 12个K线
            '1w': 1.5  # 15个K线
        }

        rebound_window = int(self.rebound_window * window_multipliers.get(timeframe, 1.0))

        rebounds = []
        indices = np.where(mask)[0]

        for i in indices:
            if i + rebound_window <= len(df):
                window = df.iloc[i:i + rebound_window]
                rebound_high = window['high'].max()
                rebound_pct = (rebound_high - level) / level

                # 考虑反弹速度和持续性
                rebound_speed = rebound_pct / rebound_window

                # 检查反弹是否持续
                sustained = window['close'].iloc[-1] > level * 1.01
                sustainability_factor = 1.2 if sustained else 1.0

                rebounds.append(rebound_pct * (1 + rebound_speed) * sustainability_factor)

        if not rebounds:
            return 0.0

        # 根据时间框架调整期望反弹幅度
        expected_rebounds = {
            '1h': 0.02,  # 2%
            '4h': 0.03,  # 3%
            '1d': 0.05,  # 5%
            '3d': 0.08,  # 8%
            '1w': 0.10  # 10%
        }

        expected = expected_rebounds.get(timeframe, 0.05)
        avg_rebound = np.mean(rebounds)

        return min(avg_rebound / expected, 1.0)

    def _calculate_volume(self, df: pd.DataFrame, mask: pd.Series,
                          timeframe: str) -> float:
        """计算成交量比率（考虑时间框架）"""
        if not mask.any():
            return 0.0

        # 使用预计算的成交量移动平均
        vol_ma = df['volume_ma'].mean() if 'volume_ma' in df else df['volume'].mean()

        # 触及时的成交量
        touch_volumes = df.loc[mask, 'volume']

        if touch_volumes.empty or vol_ma == 0:
            return 0.0

        # 成交量比率
        vol_ratio = touch_volumes.mean() / vol_ma

        # 考虑成交量峰值
        vol_spikes = (touch_volumes > vol_ma * 1.5).sum() / len(touch_volumes)

        # 综合评分
        base_score = min(vol_ratio / 2, 1.0)
        spike_bonus = vol_spikes * 0.2

        return min(base_score + spike_bonus, 1.0)

    def _calculate_recency(self, df: pd.DataFrame, mask: pd.Series,
                           cluster_data: Dict, timeframe: str) -> float:
        """计算时间新近度（考虑时间框架）"""
        # 获取最近触及时间
        last_touches = []

        if mask.any():
            last_touches.append(df.loc[mask].index.max())

        if cluster_data.get('timestamps'):
            last_touches.extend(cluster_data['timestamps'])

        if not last_touches:
            return 0.0

        last_touch = max(last_touches)

        # 计算距今的K线数量
        current_time = df.index[-1]
        time_diff = current_time - last_touch

        # 根据时间框架计算衰减
        decay_periods = {
            '1h': 24 * 7,  # 1周
            '4h': 6 * 28,  # 4周
            '1d': 30,  # 30天
            '3d': 20,  # 60天
            '1w': 12  # 12周
        }

        # 将时间差转换为对应的周期数
        if timeframe == '1h':
            periods_ago = time_diff.total_seconds() / 3600
        elif timeframe == '4h':
            periods_ago = time_diff.total_seconds() / (4 * 3600)
        elif timeframe == '1d':
            periods_ago = time_diff.days
        elif timeframe == '3d':
            periods_ago = time_diff.days / 3
        else:  # 1w
            periods_ago = time_diff.days / 7

        decay_period = decay_periods.get(timeframe, 30)

        # 指数衰减
        recency = np.exp(-periods_ago / decay_period)

        return float(recency)

    def calculate_score(self, components: Dict[str, float], timeframe: str) -> float:
        """计算总分（使用时间框架特定权重）"""
        weights = self.weights_by_timeframe.get(timeframe,
                                                self.weights_by_timeframe.get('1d', {}))

        score = sum(
            components.get(key, 0) * weight
            for key, weight in weights.items()
        )

        # 时间框架权重调整
        tf_weight = config.get(f'analysis.timeframes.{timeframe}.weight', 1.0)

        return float(score * tf_weight)

    def create_support_level(self, level: float, df: pd.DataFrame,
                             cluster_data: Dict, timeframe: str) -> SupportLevel:
        """创建支撑位对象"""
        # 计算组件
        components = self.calculate_components(level, df, cluster_data, timeframe)

        # 计算分数
        score = self.calculate_score(components, timeframe)

        # 计算置信度
        confidence = self._calculate_confidence(level, df, score, cluster_data, timeframe)

        # 分类
        category = self.categorize(score)

        return SupportLevel(
            level=level,
            score=score,
            confidence=confidence,
            category=category,
            components=components,
            metadata={
                'timeframe': timeframe,
                'cluster_strength': cluster_data.get('strength', 0),
                'cluster_count': cluster_data.get('count', 0),
                'last_update': datetime.now(),
                'atr': df['atr'].iloc[-1] if 'atr' in df else None,
                'current_price': df['close'].iloc[-1]
            }
        )
