"""Producer: Binance websocket -> Kafka topic (prices) with graceful shutdown and reconnect backoff."""

import asyncio
import json
import argparse
import signal
import websockets
from aiokafka import AIOKafkaProducer


async def binance_ws_to_kafka(
    symbol: str = "btcusdt",
    timeframe: str = "1m",
    bootstrap_servers: str = "localhost:9092",
    topic: str = "prices",
):
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    # Graceful shutdown
    def _signal_handler():
        print("🛑 Producer received stop signal, shutting down gracefully...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: _signal_handler())

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
    await producer.start()

    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{timeframe}"
    backoff = 1.0

    try:
        while not stop_event.is_set():
            try:
                async with websockets.connect(url) as ws:
                    print(f"✅ Connected to {url}, producing to {bootstrap_servers}:{topic}")
                    backoff = 1.0

                    async for msg in ws:
                        if stop_event.is_set():
                            break

                        try:
                            data = json.loads(msg)
                            k = data.get("k", {})
                            if not k:
                                continue

                            # Solo enviar vela cerrada
                            if k.get("x"):
                                payload = {
                                    "symbol": symbol[:-4].upper() + "/USDT"
                                    if symbol.endswith("usdt")
                                    else symbol.upper(),
                                    "timeframe": timeframe,
                                    "ts": int(k["t"]),
                                    "o": k["o"],
                                    "h": k["h"],
                                    "l": k["l"],
                                    "c": k["c"],
                                    "v": k["v"],
                                    "source": "binance_ws",
                                }

                                encoded = json.dumps(payload).encode("utf-8")

                                if len(encoded) > 1_000_000:  # 1 MB safeguard
                                    print("⚠️ Skipping oversized message:", len(encoded))
                                    continue

                                await producer.send_and_wait(topic, encoded)

                        except Exception as e:
                            print("⚠️ Error processing message:", e)

                await asyncio.sleep(0.1)

            except Exception as e:
                print("❌ Producer connection error:", e)
                print(f"🔄 Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    finally:
        await producer.stop()
        print("👋 Producer stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="prices")
    args = parser.parse_args()

    asyncio.run(
        binance_ws_to_kafka(
            args.symbol,
            args.timeframe,
            args.bootstrap,
            args.topic,
        )
    )
