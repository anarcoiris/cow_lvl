import pandas as pd
from datetime import datetime
import pytz

# Cargar el CSV original
df = pd.read_csv("ledger.csv")

# Convertir timestamp a datetime local (España)
df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
df["datetime"] = df["datetime"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)

# Calcular valor en USDT
df["valor_usdt"] = df["price"] * df["amount"]

# Renombrar columnas
df = df.rename(columns={
    "datetime": "Fecha y hora",
    "symbol": "instrumento",
    "side": "lado",
    "pred": "pred.",
    "price": "precio",
    "amount": "cantidad",
    "valor_usdt": "valor en USDT",
    "paper": "ejecución",
    "status": "status"
})

# Traducir lado
df["lado"] = df["lado"].map({"buy": "compra", "sell": "venta"})

# Reordenar columnas
df = df[[
    "Fecha y hora", "instrumento", "lado", "pred.",
    "precio", "cantidad", "valor en USDT",
    "ejecución", "status"
]]

# Crear nombre dinámico para el archivo
fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"ledger_{fecha_actual}.xlsx"

# Exportar a Excel con formato
with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Ledger", index=False)

    workbook  = writer.book
    worksheet = writer.sheets["Ledger"]

    # Formatos
    formato_header = workbook.add_format({
        "bold": True, "bg_color": "#4F81BD",
        "font_color": "white", "align": "center"
    })
    formato_fecha = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
    formato_numero = workbook.add_format({"num_format": "#,##0.00"})
    formato_pred = workbook.add_format({"num_format": "0.0000"})
    
    # Cabecera con color
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, formato_header)
        worksheet.set_column(col_num, col_num, 18)

    # Formatos por columna
    worksheet.set_column("A:A", 20, formato_fecha)   # Fecha y hora
    worksheet.set_column("D:D", 12, formato_pred)    # pred.
    worksheet.set_column("E:G", 15, formato_numero)  # precio, cantidad, valor USDT
    worksheet.set_column("B:C", 12)                  # instrumento, lado
    worksheet.set_column("H:I", 12)                  # ejecución, status

print(f"✅ Archivo generado: {filename}")
