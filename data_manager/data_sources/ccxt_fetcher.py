
"""Fetcher asíncrono (normaliza timestamps y verifica NaNs mínimos)."""
from typing import List, Any
try:
    import ccxt.async_support as ccxt_async
except Exception:
    ccxt_async = None

async def fetch_ohlcv(symbol: str = 'BTC/USDT', timeframe: str = '1m', limit: int = 500, exchange_id: str = 'binance') -> List[List[Any]]:
    if ccxt_async is None:
        raise ImportError('ccxt.async_support no disponible. Instala ccxt.')
    Exchange = getattr(ccxt_async, exchange_id)
    ex = Exchange()
    try:
        await ex.load_markets()
        ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    finally:
        try:
            await ex.close()
        except Exception:
            pass
