使用示例
# 启动实时扫描所有USDT交易对
python main.py

# 扫描特定交易对
python main.py -s BTC/USDT ETH/USDT BNB/USDT

# 使用自定义配置
python main.py -c config.json --log-level DEBUG

# 配置文件示例 (config.json)
{
    "monitoring": {
        "scan_interval": 30,
        "max_symbols": 100
    },
    "analysis": {
        "timeframes": {
            "15m": {"limit": 200, "weight": 0.8},
            "1h": {"limit": 120, "weight": 1.0},
            "4h": {"limit": 168, "weight": 1.5}
        }
    }
}
