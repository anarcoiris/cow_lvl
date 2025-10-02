import sqlite3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class SQLiteViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("SQLite Viewer")
        self.conn = None
        self.cursor = None

        # Botón para abrir base de datos
        self.btn_open = ttk.Button(root, text="Abrir Base de Datos", command=self.open_db)
        self.btn_open.pack(pady=5)

        # Selección de tabla
        self.table_combo = ttk.Combobox(root, state="readonly")
        self.table_combo.pack(pady=5)
        self.table_combo.bind("<<ComboboxSelected>>", self.load_table)

        # Cuadro de texto para queries
        self.query_entry = tk.Text(root, height=3, width=80)
        self.query_entry.pack(pady=5)
        self.query_entry.insert("1.0", "SELECT * FROM ...")

        # Botón ejecutar query
        self.btn_query = ttk.Button(root, text="Ejecutar Query", command=self.run_query)
        self.btn_query.pack(pady=5)

        # Tabla de resultados
        self.tree = ttk.Treeview(root, show="headings")
        self.tree.pack(fill="both", expand=True)

        # Scrollbars
        self.vsb = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.vsb.set)

    def open_db(self):
        db_file = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db *.sqlite")])
        if not db_file:
            return
        try:
            self.conn = sqlite3.connect(db_file)
            self.cursor = self.conn.cursor()
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in self.cursor.fetchall()]
            self.table_combo["values"] = tables
            if tables:
                self.table_combo.current(0)
                self.load_table()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_table(self, event=None):
        table = self.table_combo.get()
        if not table:
            return
        query = f"SELECT * FROM {table}"
        self.show_results(query)

    def run_query(self):
        query = self.query_entry.get("1.0", "end-1c").strip()
        if query:
            self.show_results(query)

    def show_results(self, query):
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            cols = [description[0] for description in self.cursor.description]

            # Reinicia TreeView
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = cols

            for col in cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120)

            for row in rows:
                self.tree.insert("", "end", values=row)
        except Exception as e:
            messagebox.showerror("Error en Query", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = SQLiteViewer(root)
    root.mainloop()
