
\"\"\"verify_coherence.py

Comprueba coherencia entre SQLite y CSV exports para un símbolo/timeframe dado.
Compara el número de filas en SQLite y en el CSV exportado más reciente, y reporta discrepancias.
También compara con processed_stats.json si existe.
\"\"\"
import os, argparse, json
import sqlite3
import pandas as pd

def count_db_rows(db_path, symbol, timeframe):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, timeframe))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else 0

def count_csv_rows(exports_dir, symbol, timeframe):
    # reads all CSV files in exports_dir and counts matching rows
    total = 0
    for fname in os.listdir(exports_dir):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(exports_dir, fname)
        try:
            df = pd.read_csv(path, usecols=['symbol','timeframe'])
            total += int((df['symbol']==symbol) & (df['timeframe']==timeframe)).sum()
        except Exception:
            continue
    return total

def read_stats(exports_dir, symbol, timeframe):
    path = os.path.join(exports_dir, 'processed_stats.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path,'r') as f:
            stats = json.load(f)
        key = f"{symbol}|{timeframe}"
        return stats.get(key, 0)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='market_data_v3.db')
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--timeframe', default='1m')
    parser.add_argument('--exports', default=os.path.join(os.path.dirname(__file__), '..', 'exports'))
    args = parser.parse_args()

    db_count = count_db_rows(args.db, args.symbol, args.timeframe)
    csv_count = count_csv_rows(args.exports, args.symbol, args.timeframe)
    stats_count = read_stats(args.exports, args.symbol, args.timeframe)

    print(f"SQLite rows for {args.symbol} {args.timeframe}: {db_count}")
    print(f"CSV exported rows for {args.symbol} {args.timeframe}: {csv_count}")
    print(f"Processed stats for {args.symbol} {args.timeframe}: {stats_count if stats_count is not None else 'N/A'}")

    if stats_count is not None:
        if db_count == stats_count == csv_count:
            print('All counts match — coherence OK.')
        else:
            print('Discrepancy detected.')
    else:
        if db_count == csv_count:
            print('DB and CSV counts match — coherence OK.')
        else:
            print('Discrepancy between DB and CSV.')
if __name__ == '__main__':
    main()
