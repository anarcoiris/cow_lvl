import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools, algorithms
from tqdm import trange
from influxdb_client import InfluxDBClient
import os

# ------------------------------
# Nueva función: leer datos desde InfluxDB
# ------------------------------
def load_data_from_influx(symbol="BTCUSDT", exchange="binance", timeframe="1m",
                          measurement="prices", field="close",
                          start="-7d", limit=1000):
    """
    Lee datos históricos desde InfluxDB y devuelve un array como simulate_ou.
    - start: periodo de tiempo que queremos traer, ej. "-7d", "-1h"
    - limit: número máximo de puntos
    """
    INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
    INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
    INFLUX_ORG = os.getenv('INFLUXDB_ORG', '')
    INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', '')

    query = f'''
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

    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        tables = client.query_api().query(query)
        values = []
        for table in tables:
            for record in table.records:
                values.append(float(record.get_value()))

    if not values:
        raise ValueError("No se obtuvieron datos de InfluxDB")

    # Convertir a formato similar a simulate_ou
    # simulate_ou retorna (paths, steps+1) → aquí usamos 1 path
    arr = np.array(values, dtype=np.float32).reshape(1, -1)
    return arr

# ------------------------------
# 1) Simulación de datos: Ornstein–Uhlenbeck *** DEPRECADO ***
# ------------------------------
def simulate_ou(theta=1.0, sigma=0.5, dt=0.01, steps=1000, paths=1000, seed=None):
    """
    Genera múltiples trayectorias de un proceso Ornstein-Uhlenbeck.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros((paths, steps + 1), dtype=np.float32)
    for t in range(steps):
        dw = np.random.normal(scale=np.sqrt(dt), size=paths).astype(np.float32)
        x[:, t+1] = x[:, t] - theta * x[:, t] * dt + sigma * dw
    return x

# ------------------------------
# 2) Preparación del dataset
# ------------------------------
def prepare_dataloaders(data, train_frac=0.8, batch_size=256):
    X = data[:, :-1].reshape(-1, 1)
    y = data[:, 1:].reshape(-1, 1)
    split = int(train_frac * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ------------------------------
# 3) Definición de la red neuronal
# ------------------------------
class Net(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = max(1, int(n_hidden))
        self.net = nn.Sequential(
            nn.Linear(1, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 4) Función de entrenamiento y evaluación
# ------------------------------
def train_and_evaluate(n_hidden, lr, train_loader, val_loader, epochs=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hidden = max(1, int(n_hidden))
    model = Net(n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total_loss += loss_fn(model(xb), yb).item() * xb.size(0)
    return total_loss / len(val_loader.dataset)

# ------------------------------
# 5) Configuración de DEAP
# ------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("n_hidden", random.randint, 5, 100)
toolbox.register("log_lr", lambda: random.uniform(-4, -1))
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.n_hidden, toolbox.log_lr),
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(ind, train_loader, val_loader):
    n_hidden, log_lr = ind
    lr = 10 ** log_lr
    mse = train_and_evaluate(n_hidden=n_hidden, lr=lr,
                             train_loader=train_loader,
                             val_loader=val_loader)
    return (mse,)

toolbox.register("evaluate", eval_individual, train_loader=None, val_loader=None)

toolbox.register("mate", tools.cxTwoPoint)
# mutación gaussiana original
toolbox.register(
    "_mutate_gauss",
    tools.mutGaussian,
    mu=[50, -2.5],
    sigma=[20, 1.0],
    indpb=0.2
)

# reparación de individuo
def repair(individual, min_hidden=5, max_hidden=200):
    individual[0] = int(round(individual[0]))
    individual[0] = max(min_hidden, min(max_hidden, individual[0]))
    individual[1] = max(-4.0, min(-1.0, individual[1]))
    return individual

# envoltorio para mutar y reparar
def mutate_and_repair(individual):
    individual, = toolbox._mutate_gauss(individual)
    return repair(individual),

toolbox.register("mutate", mutate_and_repair)
toolbox.register("select", tools.selTournament, tournsize=3)

# ------------------------------
# 6) Bucle evolutivo principal
# ------------------------------
def main_evolution(pop_size, gens, train_loader, val_loader):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_individual,
                     train_loader=train_loader,
                     val_loader=val_loader)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    for gen in trange(gens, desc="Generaciones", unit="gen"):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)
    return logbook, hof

# ------------------------------
# 7) Entrada por línea de comandos
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolución de hiperparámetros de SDE NN")
    parser.add_argument("--pop_size", type=int, default=20, help="Tamaño de población")
    parser.add_argument("--gens", type=int, default=10, help="Número de generaciones")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    data = simulate_ou(seed=args.seed)
    train_loader, val_loader = prepare_dataloaders(data)

    log, hof = main_evolution(args.pop_size, args.gens, train_loader, val_loader)

    df_log = pd.DataFrame(log)
    df_log.to_csv("evolution_log.csv", index=False)
    print("✅ Evolución guardada en evolution_log.csv")

    plt.figure()
    plt.plot(df_log["gen"], df_log["avg"], label="Fitness medio")
    plt.plot(df_log["gen"], df_log["min"], label="Fitness mínimo")
    plt.xlabel("Generación")
    plt.ylabel("MSE")
    plt.title("Evolución del fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()

    best = hof[0]
    print(f"🔥 Mejor individuo: n_hidden={int(best[0])}, lr=10^({best[1]:.2f}) => MSE={best.fitness.values[0]:.6f}")
