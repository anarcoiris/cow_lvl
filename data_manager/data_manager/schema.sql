
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT DEFAULT 'unknown',
    created_at INTEGER DEFAULT (strftime('%s','now') * 1000),
    updated_at INTEGER DEFAULT (strftime('%s','now') * 1000),
    UNIQUE(symbol, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_ts ON ohlcv(symbol, timeframe, ts);

CREATE TRIGGER IF NOT EXISTS trg_ohlcv_updated_at
AFTER UPDATE ON ohlcv
BEGIN
  UPDATE ohlcv SET updated_at = (strftime('%s','now') * 1000) WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS meta_sync (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    last_ts INTEGER,
    last_sync INTEGER
);
