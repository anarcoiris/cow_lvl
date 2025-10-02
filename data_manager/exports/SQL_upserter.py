import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox

def seleccionar_origen():
    ruta = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db *.sqlite")])
    if ruta:
        entry_origen.delete(0, tk.END)
        entry_origen.insert(0, ruta)

def seleccionar_destino():
    ruta = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db *.sqlite")])
    if ruta:
        entry_destino.delete(0, tk.END)
        entry_destino.insert(0, ruta)

def sync_bases():
    origen = entry_origen.get().strip()
    destino = entry_destino.get().strip()

    if not origen or not destino:
        messagebox.showerror("Error", "Selecciona ambas bases de datos.")
        return

    try:
        conn_origen = sqlite3.connect(origen)
        conn_destino = sqlite3.connect(destino)
        cur_origen = conn_origen.cursor()
        cur_destino = conn_destino.cursor()

        # Listar tablas
        cur_origen.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas = [row[0] for row in cur_origen.fetchall()]

        for tabla in tablas:
            cur_origen.execute(f"PRAGMA table_info({tabla})")
            cols_info = cur_origen.fetchall()
            columnas = [c[1] for c in cols_info]

            if not columnas:
                continue

            col_list = ", ".join(columnas)
            placeholders = ", ".join("?" for _ in columnas)

            # Caso especial: tabla OHLCV
            if tabla == "ohlcv":
                # clave compuesta (symbol, ts, timeframe)
                update_list = ", ".join([f"{col}=excluded.{col}" for col in columnas if col not in ("symbol","ts","timeframe")])

                cur_origen.exec ute(f"SELECT {col_list} FROM {tabla}")
                registros = cur_origen.fetchall()

                for r in registros:
                    # Aquí se fuerza la prioridad: si source=binance-ws pisa a influx
                    cur_destino.execute(f"""
                        INSERT INTO {tabla} ({col_list})
                        VALUES ({placeholders})
                        ON CONFLICT(symbol, ts, timeframe, id) DO UPDATE SET
                            {update_list}
                        WHERE excluded.source='binance-ws' OR {tabla}.source!='binance-ws'
                    """, r)

            else:
                # Caso general (PK simple en la primera columna)
                update_list = ", ".join([f"{col}=excluded.{col}" for col in columnas[1:]])

                cur_origen.execute(f"SELECT {col_list} FROM {tabla}")
                registros = cur_origen.fetchall()

                for r in registros:
                    cur_destino.execute(f"""
                        INSERT INTO {tabla} ({col_list})
                        VALUES ({placeholders})
                        ON CONFLICT({columnas[0]}) DO UPDATE SET
                            {update_list}
                    """, r)

        conn_destino.commit()
        conn_origen.close()
        conn_destino.close()
        messagebox.showinfo("Éxito", "Sincronización completada ✅")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# --- GUI ---
root = tk.Tk()
root.title("SQLite Sync (Upsert)")
root.geometry("500x200")

tk.Label(root, text="Base Origen:").pack(pady=5)
entry_origen = tk.Entry(root, width=60)
entry_origen.pack()
tk.Button(root, text="Seleccionar...", command=seleccionar_origen).pack(pady=5)

tk.Label(root, text="Base Destino:").pack(pady=5)
entry_destino = tk.Entry(root, width=60)
entry_destino.pack()
tk.Button(root, text="Seleccionar...", command=seleccionar_destino).pack(pady=5)

tk.Button(root, text="Sincronizar", command=sync_bases, bg="green", fg="white").pack(pady=15)

root.mainloop()
