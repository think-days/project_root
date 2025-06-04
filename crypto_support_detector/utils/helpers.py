"""utils/helpers.py - 辅助函数"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


def calculate_timeframe_seconds(timeframe: str) -> int:
    """计算时间框架对应的秒数"""
    multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }

    # 提取数字和单位
    import re
    match = re.match(r'(\d+)([mhdw])', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    value = int(match.group(1))
    unit = match.group(2)

    return value * multipliers[unit]


def resample_to_higher_timeframe(df: pd.DataFrame,
                                 from_tf: str,
                                 to_tf: str) -> pd.DataFrame:
    """重采样到更高时间框架"""
    from_seconds = calculate_timeframe_seconds(from_tf)
    to_seconds = calculate_timeframe_seconds(to_tf)

    if to_seconds <= from_seconds:
        raise ValueError("Target timeframe must be higher than source")

    # Pandas重采样规则
    rule_map = {
        3600: 'H',  # 1小时
        14400: '4H',  # 4小时
        86400: 'D',  # 1天
        259200: '3D',  # 3天
        604800: 'W'  # 1周
    }

    rule = rule_map.get(to_seconds)
    if not rule:
        rule = f'{to_seconds}S'

    # 重采样
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


def detect_price_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """检测价格模式"""
    patterns = {}

    # 检测趋势
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()

    if len(df) >= 50:
        current_price = df['close'].iloc[-1]
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            patterns['trend'] = 'bullish'
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            patterns['trend'] = 'bearish'
        else:
            patterns['trend'] = 'neutral'

    # 检测支撑/阻力翻转
    recent_low = df['low'].tail(20).min()
    recent_high = df['high'].tail(20).max()
    current = df['close'].iloc[-1]

    if (current - recent_low) / recent_low < 0.02:
        patterns['near_support'] = True
    if (recent_high - current) / current < 0.02:
        patterns['near_resistance'] = True

    return patterns


def calculate_risk_metrics(entry: float,
                           stop_loss: float,
                           take_profit: float,
                           position_size: float = 1.0) -> Dict[str, float]:
    """计算风险指标"""
    risk = abs(entry - stop_loss) * position_size
    reward = abs(take_profit - entry) * position_size
    risk_reward_ratio = reward / risk if risk > 0 else 0

    return {
        'risk': risk,
        'reward': reward,
        'risk_reward_ratio': risk_reward_ratio,
        'risk_pct': abs(entry - stop_loss) / entry,
        'reward_pct': abs(take_profit - entry) / entry
    }


def merge_multi_timeframe_levels(levels_by_tf: Dict[str, List]) -> List[Dict]:
    """合并多时间框架的支撑位"""
    merged = []
    tolerance = 0.005  # 0.5%容差

    # 收集所有级别
    all_levels = []
    for tf, levels in levels_by_tf.items():
        for level in levels:
            all_levels.append({
                'timeframe': tf,
                'level': level.level,
                'score': level.score,
                'category': level.category,
                'data': level
            })

    # 按价格排序
    all_levels.sort(key=lambda x: x['level'])

    # 合并相近的级别
    i = 0
    while i < len(all_levels):
        cluster = [all_levels[i]]
        j = i + 1

        # 找到相近的级别
        while j < len(all_levels):
            if (all_levels[j]['level'] - cluster[-1]['level']) / cluster[-1]['level'] < tolerance:
                cluster.append(all_levels[j])
                j += 1
            else:
                break

        # 合并聚类
        if len(cluster) > 1:
            # 计算加权平均
            total_weight = sum(c['score'] for c in cluster)
            weighted_level = sum(c['level'] * c['score'] for c in cluster) / total_weight

            merged.append({
                'level': weighted_level,
                'score': max(c['score'] for c in cluster),
                'timeframes': list(set(c['timeframe'] for c in cluster)),
                'category': max((c['category'] for c in cluster),
                                key=lambda x: ['WEAK', 'FAIR', 'GOOD', 'STRONG'].index(x)),
                'confluence': len(cluster)
            })
        else:
            merged.append({
                'level': cluster[0]['level'],
                'score': cluster[0]['score'],
                'timeframes': [cluster[0]['timeframe']],
                'category': cluster[0]['category'],
                'confluence': 1
            })

        i = j

    return merged
