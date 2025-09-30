# fix_docstrings.py
import os
import io

ROOT = os.path.abspath(os.path.dirname(__file__))

def fix_file(path):
    changed = False
    # Leer bytes para detectar bom
    with open(path, 'rb') as f:
        data = f.read()
    # Eliminar BOM UTF-8 si existe
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
        changed = True
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        print(f"Skipping non-utf8 file: {path}")
        return False

    # Reemplazos seguros: """ -> """
    # (también manejamos ''' por si acaso)
    new_text = text.replace('\\"\"\"', '"""').replace("\\'\\'\\'", "'''")

    # También, si por accidente hay un backslash justo antes de triple quotes without escaping, limpiarlo
    new_text = new_text.replace('\\\n"""', '\n"""')

    if new_text != text:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new_text)
        changed = True

    if changed:
        print(f"Fixed: {path}")
    return changed

def scan_and_fix(root):
    for dirpath, dirs, files in os.walk(root):
        # opcional: excluir venv, .git, __pycache__
        if any(x in dirpath for x in ('\\venv\\', '\\.git\\', '\\__pycache__\\')):
            continue
        for fname in files:
            if not fname.endswith('.py'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                fix_file(path)
            except Exception as e:
                print("Error fixing", path, e)

if __name__ == '__main__':
    print("Running fixer in:", ROOT)
    scan_and_fix(ROOT)
    print("Done.")
