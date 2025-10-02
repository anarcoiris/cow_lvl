#!/usr/bin/env python3
"""
deap_lstm_full_influx.py

- Lee datos de InfluxDB (1m) para un símbolo.
- Calcula features multitimeframe (1m,5m,15m,60m).
- Construye secuencias y pesos (recencia + volumen).
- Busca hiperparámetros con DEAP para una LSTM simple.
- Guarda resultados en InfluxDB (o fallback CSV) y escribe CSV / gráfica.
"""

import os
import argparse
import random
import math
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# DEAP
from deap import base, creator, tools, algorithms

# Influx
try:
    from influxdb_client import InfluxDBClient, Point
except Exception as e:
    raise ImportError("Necesitas instalar influxdb-client: pip install influxdb-client") from e

# ---------------------------
# Utilities: indicators
# ---------------------------
def compute_rsi(series, period=14):
    delta = series.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features_multi_timeframe(df, add_timeframes=[5,15,60]):
    """
    df: DataFrame with index = datetime (ascending), columns: ['close','volume']
    returns df_features with base features + aggregated features for other timeframes
    """
    df = df.copy()
    df['log_close'] = np.log(df['close'])
    df['log_return'] = df['log_close'].diff().fillna(0)
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['vola_20'] = df['log_return'].rolling(20).std()
    df['rsi_14'] = compute_rsi(df['close'], period=14)
    df = df.ffill().bfill()

    # Multi-timeframe aggregation: resample and merge back (alignment by current timestamp)
    for tf in add_timeframes:
        # tf in minutes (use 'min' to avoid pandas FutureWarning)
        rule = f'{tf}min'
        df_tf = df[['close','volume']].resample(rule).agg({
            'close': 'last',
            'volume': 'sum'
        }).ffill().bfill()
        df_tf['tf_return'] = np.log(df_tf['close']).diff().fillna(0)
        col_close = f'close_{tf}m'
        col_vol = f'vol_{tf}m'
        col_ret = f'ret_{tf}m'
        df[col_close] = df_tf['close'].reindex(df.index, method='ffill')
        df[col_vol] = df_tf['volume'].reindex(df.index, method='ffill')
        df[col_ret] = df_tf['tf_return'].reindex(df.index, method='ffill')
    df = df.dropna()
    return df

# ---------------------------
# Influx loader
# ---------------------------
def _to_naive_datetime(obj):
    """Normaliza distintos tipos de tiempo a datetime naive (UTC)."""
    if obj is None:
        return None
    if isinstance(obj, pd.Timestamp):
        dt = obj.to_pydatetime()
    elif hasattr(obj, "to_pydatetime"):
        dt = obj.to_pydatetime()
    elif isinstance(obj, datetime):
        dt = obj
    else:
        try:
            dt = pd.to_datetime(obj).to_pydatetime()
        except Exception:
            raise TypeError(f"Tipo de tiempo inesperado: {type(obj)}")
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def load_from_influx(symbol="BTCUSDT", exchange="binance", timeframe="1m",
                     measurement="prices", field="close", start="-7d", limit=20000):
    """
    Lee 'close' y 'volume' desde InfluxDB y devuelve DataFrame indexado por tiempo.
    Usa variables de entorno INFLUXDB_URL/TOKEN/ORG/BUCKET si están presentes.
    """
    INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', 'J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q==')
    INFLUX_ORG  = os.getenv('INFLUXDB_ORG', 'BreOrganization')
    INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', 'prices')

    q = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["symbol"] == "{symbol}")
  |> filter(fn: (r) => r["exchange"] == "{exchange}")
  |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  |> sort(columns: ["_time"], desc: false)
  |> limit(n: {limit})
'''
    client = None
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        tables = query_api.query(q)

        times = []
        vals = []
        for table in tables:
            for record in table.records:
                t_raw = record.get_time()
                t = _to_naive_datetime(t_raw)
                if t is None:
                    continue
                times.append(t)
                vals.append(float(record.get_value()))

        if not vals:
            raise RuntimeError("No se obtuvieron valores 'close' desde InfluxDB (consulta vacía).")

        df = pd.DataFrame({'close': vals}, index=pd.DatetimeIndex(times))
        df.index.name = "time"

        # volume query attempt
        try:
            q_vol = q.replace(f'"{field}"', '"volume"')
            tables_vol = client.query_api().query(q_vol)
            times_vol = []
            vol_vals = []
            for table in tables_vol:
                for record in table.records:
                    t_raw = record.get_time()
                    t = _to_naive_datetime(t_raw)
                    if t is None:
                        continue
                    times_vol.append(t)
                    vol_vals.append(float(record.get_value()))
            if vol_vals:
                df_vol = pd.DataFrame({'volume': vol_vals}, index=pd.DatetimeIndex(times_vol))
                df = df.join(df_vol, how='left')
                df['volume'] = df['volume'].ffill().bfill().fillna(1.0)
            else:
                df['volume'] = 1.0
        except Exception:
            df['volume'] = 1.0

        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df

    except Exception as e:
        raise RuntimeError(f"Error leyendo InfluxDB: {e}") from e

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

# ---------------------------
# Sequences builder (corrección para NumPy 2.0)
# ---------------------------
def build_sequences(df, feature_cols, window=32, predict_horizon=1, recency_tau=None, vol_alpha=0.5):
    """
    Build X,y and sample weights.
    - df: DataFrame sorted ascending index
    - feature_cols: list of columns to use
    - window: sequence length
    - predict_horizon: how many steps ahead (1 -> next step)
    - recency_tau: controls decay; if None set to window*5
    - vol_alpha: how much volume contributes to weight (0..1)
    Returns: X (N, window, features), y (N,1), weights (N,)
    """
    arr = df[feature_cols].values.astype(np.float32)
    closes = df['close'].values.astype(np.float32)
    vols = df['volume'].values.astype(np.float32)

    N = len(df) - window - predict_horizon + 1
    if N <= 0:
        raise ValueError("Not enough data for given window/predict_horizon.")

    X = np.zeros((N, window, len(feature_cols)), dtype=np.float32)
    y = np.zeros((N, 1), dtype=np.float32)
    sample_vol = np.zeros(N, dtype=np.float32)

    for i in range(N):
        X[i] = arr[i:i+window]
        y[i, 0] = closes[i+window+predict_horizon-1]
        sample_vol[i] = vols[i+window-1]  # use last vol in window

    # recency weights, so recent samples get weight ~1, older <1
    if recency_tau is None:
        recency_tau = float(window) * 5.0
    idx = np.arange(N)
    recency_weight = np.exp((idx - (N-1)) / recency_tau)  # last = 1.0

    # volume normalized 0..1 using np.ptp (compatible con NumPy 2.0+)
    vol_range = float(np.ptp(sample_vol))  # np.ptp avoids sample_vol.ptp() removal
    if vol_range <= 1e-12:
        vol_norm = np.zeros_like(sample_vol)
    else:
        vol_norm = (sample_vol - sample_vol.min()) / vol_range

    vol_weight = 1.0 + vol_alpha * vol_norm  # between 1 and 1+vol_alpha
    weights = recency_weight * vol_weight

    # normalize weights to mean 1 (avoid scaling issues)
    weights = weights / (np.mean(weights) + 1e-12)
    return X, y, weights

# ---------------------------
# LSTM model
# ---------------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

# ---------------------------
# Train / Eval
# ---------------------------
def train_eval_lstm(X_train, y_train, w_train, X_val, y_val, w_val,
                    input_size, hidden_size, lr=1e-3, epochs=5, batch_size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(input_size=input_size, hidden_size=hidden_size).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(w_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(w_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss_elems = loss_fn(preds, yb)
            loss = (loss_elems.squeeze() * wb).mean()
            loss.backward()
            opt.step()

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        tot = 0
        for xb, yb, wb in val_loader:
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            preds = model(xb)
            loss_elems = loss_fn(preds, yb)
            batch_loss = (loss_elems.squeeze() * wb).sum().item()
            total_loss += batch_loss
            tot += xb.size(0)
    val_mse = total_loss / (tot + 1e-12)
    return val_mse, model

# ---------------------------
# Save results to Influx (or fallback CSV)
# ---------------------------
def save_result_influx(influx_url, influx_token, influx_org, influx_bucket,
                       symbol, hyperparams, val_mse, notes=""):
    try:
        client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
        write_api = client.write_api()
        p = Point("deap_lstm_results") \
            .tag("symbol", symbol) \
            .field("val_mse", float(val_mse)) \
            .field("hyperparams", json.dumps(hyperparams)) \
            .field("notes", str(notes)) \
            .time(datetime.utcnow())
        write_api.write(bucket=influx_bucket, org=influx_org, record=p)
        # close
        try:
            write_api.__del__()
        except Exception:
            pass
        client.close()
        return True
    except Exception as e:
        fallback = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'hyperparams': json.dumps(hyperparams),
            'val_mse': float(val_mse),
            'notes': str(notes),
            'error': str(e)
        }
        csv_file = "deap_results_fallback.csv"
        df_fb = pd.DataFrame([fallback])
        header = not os.path.exists(csv_file)
        df_fb.to_csv(csv_file, mode='a', header=header, index=False)
        print("Warning: no se pudo guardar en InfluxDB, guardado en CSV fallback. Error:", e)
        return False

# ---------------------------
# DEAP setup
# ---------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("hidden_size_gene", random.randint, 8, 256)
toolbox.register("log_lr_gene", lambda: random.uniform(-5, -2))
toolbox.register("window_gene", random.randint, 8, 200)
toolbox.register("epochs_gene", random.randint, 1, 8)

def init_individual():
    return creator.Individual([toolbox.hidden_size_gene(),
                               toolbox.log_lr_gene(),
                               toolbox.window_gene(),
                               toolbox.epochs_gene()])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutate_ind(individual, mu=None, sigma=None, indpb=0.2):
    if mu is None:
        mu = [100, -3.5, 64, 3]
    if sigma is None:
        sigma = [50, 0.8, 30, 2]
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = individual[i] + random.gauss(0.0, sigma[i])
    individual[0] = int(max(8, min(512, round(individual[0]))))
    individual[1] = float(max(-6.0, min(-1.0, individual[1])))
    individual[2] = int(max(4, min(400, round(individual[2]))))
    individual[3] = int(max(1, min(20, round(individual[3]))))
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_ind)
toolbox.register("select", tools.selTournament, tournsize=3)

# evaluator factory
def eval_individual_factory(df_features, feature_cols, predict_horizon=1, val_fraction=0.2,
                            batch_size=256, device=None, symbol=""):
    def evaluate(individual):
        hidden_size, log_lr, window, epochs = individual
        lr = 10.0 ** log_lr
        hidden_size = int(hidden_size); window = int(window); epochs = int(epochs)

        try:
            X, y, w = build_sequences(df_features, feature_cols, window=window, predict_horizon=predict_horizon)
        except Exception as e:
            print(f"Invalid for individual {individual}: {e}")
            return (1e6,)

        N = X.shape[0]
        split = int((1.0 - val_fraction) * N)
        X_train, y_train, w_train = X[:split], y[:split], w[:split]
        X_val, y_val, w_val = X[split:], y[split:], w[split:]

        nsamples, seq, nf = X_train.shape
        X_train_resh = X_train.reshape(nsamples*seq, nf)
        mean = X_train_resh.mean(axis=0)
        std = X_train_resh.std(axis=0) + 1e-12
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        try:
            val_mse, model = train_eval_lstm(X_train, y_train, w_train, X_val, y_val, w_val,
                                             input_size=len(feature_cols),
                                             hidden_size=hidden_size,
                                             lr=lr,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             device=device)
        except Exception as e:
            print("Error during training for individual", individual, "->", e)
            return (1e6,)

        try:
            hyp = {'hidden_size': int(hidden_size), 'lr': float(lr),
                   'window': int(window), 'epochs': int(epochs)}
            INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
            INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', 'J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q==')
            INFLUX_ORG  = os.getenv('INFLUXDB_ORG', 'BreOrganization')
            INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', 'prices')
            save_result_influx(INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET,
                               symbol, hyp, float(val_mse), notes="deap_eval")
        except Exception as e:
            print("Warning: no se pudo guardar en Influx/CSV:", e)

        return (float(val_mse),)
    return evaluate

# ---------------------------
# Evolution main
# ---------------------------
def run_evolution(df_features, feature_cols, symbol, pop_size=40, gens=12, seed=None,
                  device=None, batch_size=256):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    evaluator = eval_individual_factory(df_features=df_features, feature_cols=feature_cols,
                                        predict_horizon=1, val_fraction=0.2,
                                        batch_size=batch_size, device=device, symbol=symbol)
    if "evaluate" in toolbox.__dict__:
        toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluator)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("std", np.std)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    for gen in range(gens):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        elites = tools.selBest(pop, k=max(1, int(0.02 * len(pop))))
        pop = toolbox.select(offspring + elites, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        best = hof[0]
        print(f"Gen {gen} | avg={record['avg']:.6f} min={record['min']:.6f} std={record['std']:.6f} | best={best} mse={best.fitness.values[0]:.6f}")

    return logbook, hof

# ---------------------------
# CLI / main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="-7d")
    p.add_argument("--pop_size", type=int, default=40)
    p.add_argument("--gens", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=10000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Cargando datos desde InfluxDB...")
    df = load_from_influx(symbol=args.symbol, start=args.start, limit=args.limit)
    print("Datos cargados:", len(df), "filas")

    print("Construyendo features multi-timeframe...")
    df_feat = compute_features_multi_timeframe(df, add_timeframes=[5,15,60])
    feature_cols = ['close','volume','log_return','sma_10','sma_50','vola_20','rsi_14',
                    'close_5m','vol_5m','ret_5m',
                    'close_15m','vol_15m','ret_15m',
                    'close_60m','vol_60m','ret_60m']
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    print("Feature cols used:", feature_cols)

    logbook, hof = run_evolution(df_feat, feature_cols, symbol=args.symbol,
                                 pop_size=args.pop_size, gens=args.gens, seed=args.seed,
                                 device=device, batch_size=256)

    df_log = pd.DataFrame(logbook)
    df_log.to_csv("evolution_log.csv", index=False)
    print("evolution_log.csv saved")

    plt.figure(figsize=(8,4))
    plt.plot(df_log['gen'], df_log['avg'], label='avg')
    plt.plot(df_log['gen'], df_log['min'], label='min')
    plt.xlabel('gen'); plt.ylabel('val_mse'); plt.legend(); plt.tight_layout()
    plt.savefig("evolution_plot.png")
    print("evolution_plot.png saved")

    # Save best model by re-training quick final model
    best = hof[0]
    hidden_size, log_lr, window, epochs = best
    lr = 10**log_lr
    print("Best hyperparams:", {'hidden_size':int(hidden_size),'lr':lr,'window':int(window),'epochs':int(epochs)})

    X, y, w = build_sequences(df_feat, feature_cols, window=int(window))
    mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    std = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-12
    Xn = (X - mean) / std
    split = int(0.9 * Xn.shape[0])
    X_train, y_train, w_train = Xn[:split], y[:split], w[:split]
    X_val, y_val, w_val = Xn[split:], y[split:], w[split:]
    val_mse, final_model = train_eval_lstm(X_train, y_train, w_train, X_val, y_val, w_val,
                                           input_size=len(feature_cols), hidden_size=int(hidden_size),
                                           lr=lr, epochs=int(epochs), batch_size=256, device=device)

    model_path = "best_lstm.pt"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hyperparams': {'hidden_size':int(hidden_size),'lr':float(lr),'window':int(window),'epochs':int(epochs)},
        'feature_cols': feature_cols,
        'mean': mean.tolist(),
        'std': std.tolist()
    }, model_path)
    print("Saved final model to", model_path)

    # save final result to Influx (or CSV fallback)
    try:
        hyp_final = {'hidden_size':int(hidden_size),'lr':float(lr),'window':int(window),'epochs':int(epochs)}
        INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', 'J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q==')
        INFLUX_ORG  = os.getenv('INFLUXDB_ORG', 'BreOrganization')
        INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', 'prices')
        save_result_influx(INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET,
                           args.symbol, hyp_final, float(val_mse), notes="final_model")
        print("Final result saved to Influx (or CSV fallback).")
    except Exception as e:
        print("Could not save final result to Influx/CSV:", e)

if __name__ == "__main__":
    main()
