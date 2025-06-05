import pandas as pd
import numpy as np
from crypto_support_detector.analysis.clustering import SupportLevelClustering


def make_test_df():
    periods = 60
    index = pd.date_range("2021-01-01", periods=periods, freq="H")
    # simple oscillating prices to ensure pivots
    base = np.sin(np.linspace(0, 6 * np.pi, periods)) * 10 + 100
    data = {
        'open': base + np.random.rand(periods),
        'high': base + np.random.rand(periods) + 1,
        'low': base - np.random.rand(periods) - 1,
        'close': base + np.random.rand(periods),
        'volume': np.random.rand(periods) * 100
    }
    df = pd.DataFrame(data, index=index)
    return df


def test_cluster_levels_does_not_modify_dataframe():
    df = make_test_df()
    original = df.copy(deep=True)
    clustering = SupportLevelClustering()
    clustering.cluster_levels(df)
    # Ensure no temporary columns were left behind
    assert 'tr' not in df.columns
    # Ensure original data unchanged
    pd.testing.assert_frame_equal(df, original)

