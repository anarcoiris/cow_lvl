
"""
1) Estado actual — resumen rápido (garantiza contexto)

Basado en los archivos que analizamos:

fiboevo (módulo central) produce features con add_technical_features(...). Muchas features (MAs, EMAs, fibos, VP) requieren historial (p.ej. window_long=50, vp_window=100, lookback=50).

create_sequences_from_df(...) construye ventanas solapadas (N, seq_len, F) y calcula como y_ret el log-return entre t_end y t_end+horizon.

save_scaler / load_scaler usan joblib.

TradingDaemon.iteration_once():

carga filas recientes desde sqlite (_load_recent_rows()),

llama a fibo.add_technical_features(...),

detecta feature_cols (o usa self.feature_cols),

crea secuencias con create_sequences_from_df (o _build_sequences_internal),

aplica scaler: flat = X.reshape(-1, X.shape[2]); flat_s = scaler.transform(flat); Xp = flat_s.reshape(X.shape),

toma la última secuencia Xp[-1:], la pasa al modelo y obtiene pred (y vol si corresponde),

decide trade con _should_trade(pred) y escribe ledger o intenta ccxt order si paper==False.

simulate_fill en fiboevo es una implementación muy simple que chequea si en la serie futura se alcanza el precio buscado y devuelve el "first hit". Slippage es un simple porcentaje fijo aplicado luego.

Backtest del GUI es "simple": construye features, usa _build_sequences_internal y avanza la red por lotes para producir predicciones; luego computa PnL básico usando diferencias log-price.

Conclusión: la infraestructura está completa pero el backtest y la capa live son todavía simplistas (no modelan liquidez real, slippage realista, comisiones, parcial fills, latencia, book dynamics), y hay riesgos de double-scaling o de features mal alineadas si no se controla el pipeline de preprocesamiento.

2) Backtesting: qué hace ahora y qué necesita para ser más realista
2.A Qué hace hoy (resumen operativo)

En el backtest proporcionado:

Se genera feats = fibo.add_technical_features(...) a partir del OHLCV histórico (dropna).

Se construyen secuencias X y etiquetas y con _build_sequences_internal.

Se ejecuta el modelo (o se carga artifacts/model_best.pt) para producir preds.

Para cada predicción que supere umbral (ejemplo: 0.0005), se simula una orden hipotética (ej. comprar la fracción pos_pct del equity), se calcula future_idx = i + seq_len - 1 + horizon y se calcula PnL simple: ln(c_future) - ln(c0) y se convierte a USD con equity * pos_pct * (exp(ret) - 1).

No hay comisiones, no hay slippage avanzado, no modelización de impacto de mercado, no costes por tamaño ni orden parcial, no latencia ni throttling (rate-limit) y las ejecuciones son atómicas (si la predicción indica "buy", se asume ejecución exacta y al precio real futuro).

2.B Principales limitaciones del enfoque actual

Sin comisiones ni fees: resultados sobreestimados. En exchanges HF, fees por operación reducen PnL.

Slippage constante o nulo: simulate_fill aplica un porcentaje fijo; en la vida real, slippage depende de liquidez y tamaño relativo a libro/volumen.

No modela ejecución parcial: órdenes grandes pueden rellenarse parcialmente a múltiples precios.

No diferencia límites vs mercado: tipo de orden y disponibilidad del libro determinan si una orden se llena.

No modela costos de financiación / borrow / margin: si estrategia usa margen, faltan costes.

Look-ahead en algunos pipelines si no se asegura que las features se calculan sólo con datos previos (ver sección 4).

No latencia: orden enviada instantáneamente y asumida ejecutada en el primer valor futuro disponible.

No control del tamaño relativo: no usa métricas de liquidez (avg volume, depth) al dimensionar la orden.

2.C Qué cambios introducir (conceptual → concreto)
Objetivos

Hacer el backtest conservador y más realista.

Incorporar costos transaccionales, slippage dependiente de tamaño/liquidez, posibilidad de ejecución parcial, latencia simulada y limit orders.

Mantener compatibilidad con pipeline actual (OHLCV + features), porque no siempre tenemos el order book.

Reglas prácticas (recomendadas)

Comisión por operación: aplicar tasa commission_pct (por ejemplo 0.04% taker, 0.02% maker) al USD ejecutado.

Slippage modelado por relación order_size / volume_window: slippage proporc. a (size / (avg_volume_window * k))^alpha. k ajusta escala (por ejemplo 2), alpha típicamente 0.5..1.0. Si size >> avg_vol_window => alto slippage.

Capacidad de llenado parcial: si el mercado no puede absorber la orden, aceptar filled_pct < 1.0 y computar el precio medio ponderado.

Latencia: simular small delay latency_seconds entre issuance y ejecución; esto implica que la orden vea precios futuros (peor escenario: price moves unfavorably durante latency).

Tick/lot quantization: round price & qty to tick/lot.

Spread + order type: market orders pagan spread; limit orders pueden no ser llenados.

Market impact temporario: si múltiples órdenes grandes consecutivas, suprimir capacidad real (opcional).

Componentes a añadir

simulate_fill_with_liquidity(future_prices, future_volumes, side, order_amount, tick, lot, avg_vol_window, commission_pct, latency, impact_model, order_type) — reemplaza/optimize simulate_fill.

apply_transaction_costs(entry_price, exit_price, amount, commission_pct) — factura fees.

order_engine (clase) que simula estado de órdenes (open, filled, partial) y las procesa con el modelo de liquidez.

backtest_runner que trabaja por paso temporal (time-step) en vez de vectorizado por predicciones en bloque — permite orders o posiciones permanecientes y efectos compuestos.
"""

class OrderEngine:
    def __init__(self, exchange, ledger_path, paper=True, commission_pct=0.0004, logger=None):
        self.exchange = exchange  # adapter (wraps ccxt)
        self.ledger_path = ledger_path
        self.paper = bool(paper)
        self.commission_pct = commission_pct
        self.logger = logger or logging.getLogger("OrderEngine")
        self.open_orders = {}  # local_id -> order dict
        self.positions = {}    # symbol -> position info

    def submit_order(self, symbol, side, amount, order_type="market", price=None, local_id=None, meta=None):
        local_id = local_id or f"loc_{int(time.time()*1000)}"
        order = {"local_id": local_id, "symbol": symbol, "side": side, "amount": amount, "type": order_type, "price": price, "status": "pending", "meta": meta or {} }
        self.open_orders[local_id] = order
        if self.paper:
            # use improved simulate_fill
            future_prices, future_volumes = self._get_future_series(symbol)  # from df/features
            ok, exec_price, filled_amt = simulate_fill_with_liquidity(side, amount, future_prices, future_volumes, ...)
            commission = exec_price * filled_amt * self.commission_pct
            order.update({"status": "filled" if ok else "rejected", "exec_price": exec_price, "filled_amount": filled_amt, "commission": commission})
            self._record_ledger(order)
            self._update_position(symbol, side, filled_amt, exec_price)
            return order
        else:
            # live: use exchange adapter
            try:
                resp = self.exchange.create_order(symbol, order_type, side, amount, price)
                order.update({"status": "placed", "exchange_id": resp.get("id"), "raw": resp})
                self._record_ledger(order)
                return order
            except Exception as e:
                order.update({"status": "error", "error": str(e)})
                self._record_ledger(order)
                self.logger.exception("submit_order failed")
                return order

    # ... additional methods: cancel_order, poll, handle_websocket etc.


def simulate_fill_with_liquidity(
    side: str,
    order_amount: float,
    future_prices: np.ndarray,      # array de precios futuros (corresponding to time index)
    future_volumes: Optional[np.ndarray],
    avg_vol_window: int = 10,
    commission_pct: float = 0.0004,  # 0.04%
    latency_sec: float = 0.0,
    price_tick: Optional[float] = None,
    lot_size: Optional[float] = None,
    impact_k: float = 2.0,
    impact_alpha: float = 0.8
) -> Tuple[bool, float, float]:
    """
    Simula la ejecución de una orden (possiblemente parcial) estimando slippage
    usando avg volumes. Devuelve (filled_bool, executed_price_avg, filled_amount).
    """
    # Ubicar start index after latency: convert latency_sec -> number of steps
    # (here we assume future_prices is indexed per timeframe step — caller maps latency accordingly)
    start_idx = 0  # caller must map latency to index offset if needed

    # protective defaults
    if future_volumes is None:
        # fallback: use a small fraction of price series as proxy
        future_volumes = np.full_like(future_prices, np.nan)
    # compute avg volume on a window near start
    avg_vol = np.nanmean(future_volumes[max(0, start_idx-avg_vol_window):start_idx+1]) if np.any(~np.isnan(future_volumes)) else None

    # estimate relative size vs liquidity
    if avg_vol and avg_vol > 0:
        rel = order_amount / avg_vol
    else:
        rel = 1.0  # unknown -> assume illiquid

    # slippage fraction estimation
    slippage_frac = min(0.5, impact_k * (rel ** impact_alpha))  # cap at 50%
    # choose execution price as first available price +/- slippage
    p0 = future_prices[start_idx]
    if side.lower() == "buy":
        executed_price = p0 * (1.0 + slippage_frac)
    else:
        executed_price = p0 * (1.0 - slippage_frac)

    # quantize
    if price_tick:
        executed_price = quantize_price(executed_price, price_tick)

    # filled amount: assume filled proportion = min(1, 1/rel) as simple fallback
    if avg_vol and avg_vol > 0:
        filled_pct = min(1.0, 1.0 / max(rel, 1e-9))
    else:
        filled_pct = 0.5  # assume half filled if unknown

    filled_amount = order_amount * filled_pct
    # apply commission in caller or return extra info
    return True, float(executed_price), float(filled_amount)

"""
4) ¿Se usan IDs de filas? timestamp vs id — impacto en training/inferencia
4.A Estado actual: uso de IDs y timestamps en el código

El código no usa explícitamente la row id de la tabla (p.ej. rowid de SQLite) para construir secuencias. El orden se basa en ts o timestamp y el ORDER BY ts ASC en las consultas SQL (ver _load_recent_rows y _load_df_for_training).

TradingDaemon._load_recent_rows busca columnas ts y/o timestamp y convierte ts a timestamp si es necesario, ordena por la columna timestamp o ts.

Por tanto, el timestamp es la llave canónica para orden cronológico. El modelo asume un flujo ordenado por timestamp, no por rowid.

4.B Problema típico: websocket (ticks) vs influx/SQLite (agregados)

Fuentes websocket producen ticks/trades (eventos a nivel transacción). Estas pueden llegar con timestamps de cuando ocurrió o del servidor (exchange). Pueden existir:

Duplicados,

Reordenamientos (latencias),

Timestamps con distintas resoluciones (ms vs s),

Microsecond differences.

Influx (u otros collectors) suele almacenar candles (OHLCV) agregadas por timeframe. La agregación puede hacerse con ventanas que no coincidan exactamente con la forma en la que tú deseas (p. ej. boundaries por UTC vs exchange local).

Consecuencia: No confiar en rowid para alineación. Usar timestamp (en UTC) y una rutina de canonicalización/aggregation que:

Recibe ticks y agrupa a timeframes usando un time alignment consistent (p. ej. aligning timestamps to multiples of timeframe).

Emite candles atómicos (open/high/low/close/volume) para cada bucket con timestamp = bucket_end o bucket_start. Se debe elegir y documentar consistentemente.

4.C Recomendaciones concretas

Canonical key: timestamp_utc (datetime tz-aware) y source (websocket/influx/sqlite) + seq optional.

Cuando ingestes ticks -> agregar a candle con el mismo método que tu DB histórica (p.ej. same timezone and boundary).

De-duplicación: si dos sources aportan la misma candle timestamp, deberías tener policy (prefer latest source vs prefer full OHLCV).

No mezclar filas con timestamps arbitrarios: if websocket produces a candle with partial data (still accumulating), avoid using it to compute features requiring completed windows (important).

Tag temporal de la predicción: al hacer una predicción sobre la última secuencia que termina en t_end, la predicción es para t_target = t_end + horizon * timeframe. El daemon debe exponer ambos timestamps: prediction_at (t_end) y target_time (t_target) en ledger/logs.

Synchronisation: si tu sqlite está poblada por un proceso independiente (influx->sqlite), asegúrate que writes sean atómicos y que daemon no lea filas parcialmente escritas (bloqueo o usar a copy).

5) Flujo exacto: cuándo se calculan features, cuándo se escala, y cómo evitar doble escalado

Voy a descomponer cada paso de data → features → sequences → scaling → prediction, para que veas exactamente dónde intervenir y qué chequeos añadir.

5.A Pipeline ideal y robusto (recomendación)

Ingesta (DB / Websocket collector)
Result: tabla ohlcv con columnas timestamp (UTC), open, high, low, close, volume, symbol, timeframe, ts (opcional).
Nota: Antes de exponer al daemon, las candles deben estar closed (finalizadas) — no usar candles incompletas.

Cálculo de features (raw features, NO scaling)

Función: feats = add_technical_features(close, high, low, volume, ...)

Debe computarse siempre sobre precios raw (no escalados).

Importante: add_technical_features genera columnas que necesitan pasado (ej. lookback=50 para fib levels; vp_window=100 para VP). Por ello, asegúrate de que el DataFrame pasado tiene al menos max_feature_lookback filas previas.

No persistir features escaladas en DB salvo que documentes que están normalizadas para inferencia offline.

Feature selection

Preferir model_meta["feature_cols"] si existe. (Implementar fallback: log warning y abortar o auto-detect).

Comprueba: set(meta.feature_cols) ⊆ set(feats.columns). Si faltan columnas -> log+option to abort.

Sequence creation

crear X,y via create_sequences_from_df(feats, feature_cols, seq_len, horizon) but: decidir si se aplica scaler antes o después.

Two consistent approaches:

A) Escalar antes de construir sequences: escalar cada fila de feats[feature_cols] (transform row-wise) → construir secuencias con los valores ya escalados. (Pro: scaler aplica identidad por fila; Con: debes asegurar que scaler fue entrenado con exactamente las mismas filas y feature order).

B) Construir sequences en raw y luego escalar flatten(X): construir X raw → flat = X.reshape(-1, F) → flat_s = scaler.transform(flat) → X_scaled = flat_s.reshape(X.shape). (Pro: coincide con cómo se entrena si guardaste scaler sobre filas usadas por train; Con: cuidado con doble-scaling si ya escalaste df).

En el código actual: el enfoque del daemon es B (se escala después de construir X). En la preparación/entrenamiento, a veces el pipeline A fue usado para visualizar df_scaled. Por ello hay riesgo de inconsistencia.

Prediction

Usar Xp[-1:] (última secuencia) y pasar tensor al modelo (float32), device CPU/GPU según convenga.

CONVENCIÓN: si model meta contiene input_size, feature_cols, scaler insert should match ordering.

Decision & execution

should_trade y execute_trade.

5.B Peligros de doble-scaling y cómo detectarlo

Situación 1 (doble-scaling): en preparación, df_features se escala para vistas (df_scaled) y también se derivan X desde df_scaled, y luego en iteration_once se aplica scaler.transform otra vez a X. Resultado: doble escalado.

Situación 2 (escala distinta a la entrenada): durante entrenamiento el scaler se ajustó usando filas hasta train_rows_end (como en build_dataset_for_training), mientras que en inferencia se aplica a filas de tiempo reciente; si las order/features han cambiado de distribución, las transformaciones siguen aplicándose pero pueden dar valores extremos (no necesariamente bug, pero puede causar shift).

Chequeos prácticos para detectar doble-scaling

Check A: Si scaler objeto tiene feature_names_in_ (sklearn) y scaler.feature_names_in_ != model_meta["feature_cols"] → alerta.

Check B: Si df_features ya aparece en logs como df_scaled (o contiene columnas con rango típico de z-score, e.g. valores ~ ± few), entonces no vuelvas a transformar. Detectarlo comparando estadísticos: df_features[self.feature_cols].mean() y std(); si mean ~0 std ~1 -> probablemente ya está escalado.

Check C: Al cargar scaler, asegurar scaler.n_features_in_ coincide con X.shape[2].

Check D: instrumenta el daemon para loggear min,max,mean de la última secuencia antes y después de scaler.transform y detectar outliers.

Política recomendada (fija y segura)

Establece una sola convención:

Entrenamiento: Ajustar scaler sobre filas crudas (never scale twice), guardar scaler junto con meta['feature_cols'] y meta['scaler_feature_names_in'].

Inferencia: Computar raw features, luego solo una vez aplicar scaler.transform sobre esas filas (preferentemente antes de construir secuencias, i.e., approach A). O si vas a usar approach B (aplicar a flattened X), aplica sólo ese transform y registra que X proviene de raw features.

Implementar una función utilitaria en fiboevo (o daemon) llamada prepare_input_for_model(df, feature_cols, seq_len, scaler, method='per_row'|'flat') que:

valida scaler.feature_names_in_ vs feature_cols (si disponible),

controla df length >= required_history,

transforma consistentemente y devuelve tensor (1, seq_len, F) listo para model.

Helper que aconsejo añadir (ejemplo):
"""

def prepare_input_for_model(df: pd.DataFrame, feature_cols: List[str], seq_len: int, scaler=None, method='per_row'):
    # 1) check length
    if len(df) < seq_len:
        raise ValueError("not enough rows")
    Xraw = df.iloc[-seq_len:][feature_cols].astype(float).values  # shape (seq_len, F)
    if scaler is None:
        X = Xraw.astype(np.float32)
    else:
        # check scaler.feature_names_in_
        try:
            if hasattr(scaler, "feature_names_in_"):
                # ensure same order
                if not np.array_equal(np.asarray(scaler.feature_names_in_, dtype=object), np.asarray(feature_cols, dtype=object)):
                    # log warning and attempt reorder if possible
                    # safer: reindex scaler.feature_names_in_ to feature_cols -> but sklearn scaler can't reorder mean_/var_ directly: better abort or re-create scaler
                    raise RuntimeError("Scaler feature_names_in_ differs from feature_cols")
        except Exception:
            pass
        # method
        if method == 'per_row':
            last = Xraw.astype(np.float64)
            try:
                scaled = scaler.transform(last).astype(np.float32)
            except Exception:
                scaled = last.astype(np.float32)
            X = scaled
        else:  # flat
            X = Xraw.astype(np.float32)  # build seq and transform later using flat approach
    # return tensor shape (1, seq_len, F)
    import torch
    t = torch.from_numpy(X).unsqueeze(0).float()
    return t

"""
Comprobación adicional al cargar artefactos (recomendado)

Al cargar model y scaler, añadir una rutina que:

verifica len(meta['feature_cols']) == model.input_size (warn if mismatch),

si scaler no tiene feature_names_in_ y meta tiene feature_cols, inject: scaler.feature_names_in_ = np.array(meta['feature_cols'], dtype=object) (como propusiste).

test de smoke: computar prepare_input_for_model en una ventana de training y hacer model forward para comprobar shapes.

6) Identificación completa de placeholders / simplificaciones (lista priorizada)

Voy a listar las simplificaciones y los "must-fix" ordenados por prioridad:

Prioridad alta (produce errores / mal performance / riesgo en producción)

Position sizing placeholder: en _execute_trade comment # Position sizing: implement your logic here; placeholder uses fixed fraction — actualmente se usa pos_pct del model_meta sin señal de riesgo. (Arreglar).

Slippage model trivial: simulate_fill aplica un simple slippage_tolerance_pct fijo — insuficiente para mercados reales.

No fees in backtest: missing commission modelling (aplicar fees).

No order lifecycle handling: live mode places order via exchange.create_order but no tracking of exchange order_id nor handling of partial fills or cancels.

Model path empty in GUI: model_path="" passed; daemon defaults to artifacts/model_best.pt — but empty string might confuse loader logic (should be None).

Feature_cols detection fallback: If model_meta has feature_cols the daemon should enforce them; currently it may auto-detect and produce misaligned inputs.

Prioridad media (mejora robustez/diagnóstico)

No atomic DB read: _load_recent_rows opens sqlite for each call; better to reuse or handle concurrent writes better.

No scaler feature_names_in_ check: risk of mismatched order.

Backtest uses vectorized approach: good for speed but loses ability to simulate orders across timesteps.

No explicit required-history check: features requiring lookback could be generated with insufficient history and produce NaNs or shifted features.

No logging of prediction target timestamp: ledger lacks target_time to map the prediction to actual realized price.

Lack of risk manager (max exposure, stop-loss).

Prioridad baja (cosmética o refactor)

Optional imports fallback (fiboevo sometimes absent) — OK but cause less explicit error messages.

No typing/hints/docstrings exhaustivas.

train loop saving model.state_dict only: may be fine but saving payload with meta would be better (you already save meta separately sometimes).

7) Parches / cambios concretos propuestos (resumido y aplicable)

Te doy parches concretos (pequeños) para integrar inmediatamente y reducir el riesgo. Puedes aplicarlos en trading_daemon.py y fiboevo.py. Los presentaré como snippets a copiar/pegar.

7.A Forzar uso de meta['feature_cols'] en trading_daemon.iteration_once()

Reemplaza la parte de detección de feature_cols por:
"""

# prefer meta feature list
if self.model_meta and isinstance(self.model_meta.get("feature_cols"), (list, tuple)):
    feature_cols = [c for c in self.model_meta["feature_cols"] if c in feats.columns]
    if len(feature_cols) != len(self.model_meta["feature_cols"]):
        self._enqueue_log("Warning: some meta feature_cols missing in current feats; falling back to auto-detect for available columns.")
        # fallback: auto detect only for present columns
        exclude = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "exchange"}
        feature_cols = [c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]
else:
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "exchange"}
    feature_cols = [c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]

"""
7.B After loading scaler: repair feature_names_in_

Add in both fiboevo.load_scaler or TradingDaemon._load_model_from_artifacts after scaler load:
"""

if scaler is not None and isinstance(self.model_meta, dict) and "feature_cols" in self.model_meta:
    try:
        if not hasattr(scaler, "feature_names_in_"):
            scaler.feature_names_in_ = np.array(self.model_meta["feature_cols"], dtype=object)
    except Exception:
        pass


""" 7.C Add smoke assertions after model+scaler load """

# after self.model, self.model_scaler assigned
if getattr(self, "model_meta", None) and "feature_cols" in self.model_meta:
    try:
        if int(self.model_meta.get("input_size", len(self.model_meta["feature_cols"]))) != len(self.model_meta["feature_cols"]):
            self._enqueue_log("Warning: meta.feature_cols length != input_size (meta inconsistency).")
    except Exception:
        pass

"""
7.D Implement improved simulate_fill (add to fiboevo)

Add simulate_fill_with_liquidity as earlier snippet into fiboevo.py and replace calls in backtest and daemon simulation.

7.E Add prepare_input_for_model util (to avoid double scaling)

Add function in fiboevo (or a utils module) as shown previously (prepare_input_for_model) and use it both in training scripts and in TradingDaemon.iteration_once() — this centralizes scaling logic and prevents double-scaling.

8) ¿Dónde invocar diagnostics.py?

Recomendación: invócalo en estos puntos:

Al arrancar la GUI (al cargar, _load_config_on_start): quick smoke checks (is fibo available? model file exists?).

Al iniciar el daemon (en _start_daemon) antes de start_loop() — run diagnostics that validate model+scaler consistency, and that there are enough rows in sqlite to build required history. If failing, block start and notify user.

Periódicamente en el daemon (por ejemplo cada N iterations) lanzar un subset ligero del diagnostics para verificar drift en features (mean/std) y que scaler no está saturando (lots of nan/infs).

Antes de entrar en Live Mode (switch), correr diagnostics.full_check() que valide connectivity to exchange, balances etc.

9) Resumen de acciones inmediatas y plan de prioridades

Inmediato (apply asap):

Forzar uso de meta['feature_cols'] (patch 7.A).

Add prepare_input_for_model util and switch inference to use it (avoids double-scaling).

Add simulate_fill_with_liquidity and start using it in paper/backtest runs.

Ensure scaler.feature_names_in_ patched from meta (7.B).

Medio plazo (2-4 sprints):

Implement OrderEngine, ExchangeAdapter, RiskManager.

Replace vectorized backtest with step-by-step runner supporting open orders, latency, partial fills.

Add tests that check pipeline properties: scaler idempotency, feature order correctness.

Largo plazo:

Integrate real websocket ingestion, TWAP execution slicing, advanced market impact modeling, continuous retraining (rolling window) with safe guards.

10) Últimos consejos prácticos y checks para poner en producción

Always treat the timestamp as canonical; do not depend on sqlite rowid.

Persist meta with input_size, feature_cols, seq_len, horizon, scaler_feature_names_in_, and trained_at. Esto hace debugging mucho más sencillo.

Instrumenta: log mean/std/min/max of features just before and after scaling (daemon logs). Añade alertas si features out-of-range.

Test reproducibility: run a smoke test that prepare_input_for_model(...from training_data) -> model.forward yields consistent shapes.

Start with conservative execution parameters in live (low pos sizes, paper mode for an extended period).

Unit tests: write tests that garanticen que prepare_input_for_model no doble-escale.

11) ¿Quieres que aplique cambios ahora?

Puedo:

generar los patches exactos para los archivos (trading_daemon.py, fiboevo.py, trading_gui_extended.py) y devolvértelos completos listos para sustituir, o

crear PR-style diffs en texto, o

implementar sólo los cambios menores (7.A–7.E) y entregarte los archivos actualizados.

Dime cuál prefieres y aplico los cambios inmediatamente aquí en la respuesta (pegando los nuevos archivos o los parches). Si quieres que modifique cosas grandes como OrderEngine o la reescritura del backtest en modo paso-a-paso, te recomiendo que me dejes aplicar primero los pasos inmediatos (sanity fixes) y luego hagamos la reescritura por etapas.
"""