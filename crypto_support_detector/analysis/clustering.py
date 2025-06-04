"""analysis/clustering.py - 聚类分析模块"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional

from ..config.settings import config
from .pivots import PivotDetector


class SupportLevelClustering:
    """支撑位聚类分析"""

    def __init__(self):
        self.pivot_detector = PivotDetector()
        self.scaler = StandardScaler()

    def adaptive_eps(self, df: pd.DataFrame) -> float:
        """自适应计算聚类半径"""
        df = df.copy()

        # 计算ATR
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        atr_pct = df['tr'].tail(20).mean() / df['close'].mean()
        df.drop(columns=['tr'], inplace=True)

        # 计算价格波动率
        volatility = df['close'].pct_change().tail(50).std()

        # 综合ATR和波动率
        base_eps = config.get('analysis.base_eps_pct', 0.003)
        dynamic_eps = max(base_eps, min(atr_pct * 0.3, volatility * 0.5))

        return dynamic_eps

    def cluster_levels(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """聚类支撑位"""
        # 获取支点
        pivots = self.pivot_detector.get_pivot_points(df)

        if pivots.empty:
            return {}

        # 计算自适应eps
        eps = self.adaptive_eps(df) * df['close'].mean()
        min_samples = config.get('analysis.cluster_min_samples', 2)

        # DBSCAN聚类
        X = pivots.values.reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # 整理聚类结果
        clusters = {}
        for label in set(clustering.labels_):
            if label == -1:  # 噪声点
                continue

            mask = clustering.labels_ == label
            cluster_values = pivots.values[mask]
            cluster_times = pivots.index[mask]

            clusters[label] = {
                'level': float(np.mean(cluster_values)),
                'std': float(np.std(cluster_values)),
                'count': len(cluster_values),
                'values': cluster_values.tolist(),
                'timestamps': cluster_times.tolist(),
                'strength': self._calculate_cluster_strength(
                    cluster_values, cluster_times, df
                )
            }

        return clusters

    def _calculate_cluster_strength(self, values: np.ndarray,
                                    times: pd.DatetimeIndex,
                                    df: pd.DataFrame) -> float:
        """计算聚类强度"""
        # 时间跨度
        time_span = (times.max() - times.min()).days

        # 价格一致性
        consistency = 1 - (np.std(values) / np.mean(values))

        # 近期活跃度
        recent_ratio = sum(times > df.index[-30]) / len(times)

        # 综合强度
        strength = (
                0.3 * min(len(values) / 10, 1) +  # 触及次数
                0.3 * consistency +  # 一致性
                0.2 * recent_ratio +  # 近期活跃
                0.2 * min(time_span / 90, 1)  # 时间跨度
        )

        return float(strength)

    def merge_nearby_clusters(self, clusters: Dict[int, Dict],
                              threshold: float = 0.005) -> Dict[int, Dict]:
        """合并相近的聚类"""
        if len(clusters) <= 1:
            return clusters

        # 获取所有聚类中心
        centers = [(k, v['level']) for k, v in clusters.items()]
        centers.sort(key=lambda x: x[1])

        # 合并相近的聚类
        merged = {}
        merged_id = 0
        i = 0

        while i < len(centers):
            current_cluster = [centers[i]]
            j = i + 1

            # 查找可合并的聚类
            while j < len(centers):
                if (centers[j][1] - current_cluster[-1][1]) / current_cluster[-1][1] < threshold:
                    current_cluster.append(centers[j])
                    j += 1
                else:
                    break

            # 合并聚类数据
            if len(current_cluster) > 1:
                merged_data = self._merge_cluster_data(
                    [clusters[c[0]] for c in current_cluster]
                )
                merged[merged_id] = merged_data
            else:
                merged[merged_id] = clusters[current_cluster[0][0]]

            merged_id += 1
            i = j

        return merged

    def _merge_cluster_data(self, cluster_list: List[Dict]) -> Dict:
        """合并多个聚类的数据"""
        all_values = []
        all_times = []

        for cluster in cluster_list:
            all_values.extend(cluster['values'])
            all_times.extend(cluster['timestamps'])

        return {
            'level': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'count': len(all_values),
            'values': all_values,
            'timestamps': all_times,
            'strength': np.mean([c['strength'] for c in cluster_list])
        }
