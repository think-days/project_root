"""config/settings.py - 配置管理"""
from typing import Dict, Any
import json
from pathlib import Path

class Config:
    """配置管理器"""

    # 更新默认配置，适应加密货币市场特性
    DEFAULT_CONFIG = {
        "data": {
            "exchange": "binance",
            "base_url": "https://api.binance.com",
            "rate_limit": 1200,
            "cache_ttl": 300,
            # 交易对过滤条件
            "symbol_filters": {
                "min_age_days": 30,        # 排除新币（30天内）
                "exclude_tags": ["monitoring", "assessment"],  # 排除观察标签
                "quote_currencies": ["USDT", "BUSD"],  # 允许的计价货币
                "exclude_stablecoin_pairs": True,  # 排除稳定币对
                "min_volume_usdt": 1000000,  # 最小24h成交量（USDT）
                "max_symbols": 200
            }
        },
        "analysis": {
            "pivot_window": 5,
            "base_eps_pct": 0.003,
            "cluster_min_samples": 2,
            "allow_margin": 0.02,
            # 针对加密货币市场优化的时间框架配置
            "timeframes": {
                "1h": {
                    "limit": 168,      # 7天数据
                    "weight": 0.8,     # 权重较低，噪音较多
                    "min_touches": 3,  # 最少触及次数
                    "lookback_days": 7
                },
                "4h": {
                    "limit": 168,      # 28天数据（4周）
                    "weight": 1.2,     # 中等权重
                    "min_touches": 3,
                    "lookback_days": 28
                },
                "1d": {
                    "limit": 90,       # 90天数据（3个月）
                    "weight": 1.5,     # 较高权重
                    "min_touches": 2,
                    "lookback_days": 90
                },
                "3d": {
                    "limit": 60,       # 180天数据（6个月）
                    "weight": 1.8,     # 高权重
                    "min_touches": 2,
                    "lookback_days": 180
                },
                "1w": {
                    "limit": 52,       # 52周数据（1年）
                    "weight": 2.0,     # 最高权重
                    "min_touches": 2,
                    "lookback_days": 365
                }
            }
        },
        "scoring": {
            # 根据时间框架动态调整的权重
            "weights": {
                "1h": {
                    "touches": 0.20,
                    "rebound": 0.30,
                    "volume": 0.25,
                    "recency": 0.25
                },
                "4h": {
                    "touches": 0.25,
                    "rebound": 0.35,
                    "volume": 0.20,
                    "recency": 0.20
                },
                "1d": {
                    "touches": 0.30,
                    "rebound": 0.35,
                    "volume": 0.20,
                    "recency": 0.15
                },
                "3d": {
                    "touches": 0.35,
                    "rebound": 0.35,
                    "volume": 0.15,
                    "recency": 0.15
                },
                "1w": {
                    "touches": 0.40,
                    "rebound": 0.30,
                    "volume": 0.15,
                    "recency": 0.15
                }
            },
            "thresholds": {
                "excellent": 0.8,
                "good": 0.6,
                "fair": 0.4,
                "poor": 0.2
            },
            "rebound_window": 10,
            "volume_ma_window": 20
        },
        "backtest": {
            "eps_pct": 0.003,
            "target_pct": 0.03,
            "look_ahead": 24
        },
        "monitoring": {
            "scan_interval": 300,     # 5分钟扫描一次
            "alert_cooldown": 3600,   # 1小时告警冷却
            "max_concurrent": 50,     # 最大并发请求
        }
    }

    def __init__(self, config_path: str = None):
        self._config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: str):
        """从文件加载配置"""
        with open(path, 'r') as f:
            custom_config = json.load(f)
        self._merge_config(custom_config)

    def _merge_config(self, custom: Dict[str, Any]):
        """递归合并配置"""
        def merge(base, custom):
            for key, value in custom.items():
                if isinstance(value, dict) and key in base:
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(self._config, custom)

    def get(self, key_path: str, default=None):
        """获取配置值，支持点号路径"""
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

# 全局配置实例
config = Config()
