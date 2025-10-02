COLOR C
python -m data_sources.influx_to_sql --mode both --symbol BTCUSDT --exchange binance --timeframe 1m --start '-90d' --bucket prices --sqlite ".\exports\1m_influx30sept25.db"