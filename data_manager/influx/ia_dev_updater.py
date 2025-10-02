#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
import os
import subprocess
from openai import OpenAI
from pathlib import Path
import re

client = OpenAI(api_key="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")

REPOS_BASE_DIR = Path(r"C:\Users\soyko\Documents\git")
BRANCH_NAME = "IA-Dev"
CODE_EXTENSIONS = {".py"}

def run_cmd(cmd, cwd=None):
    print(f"Ejecutando: {' '.join(cmd)} en {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error ejecutando {' '.join(cmd)}:\n{result.stderr}")
    return result.stdout.strip()

def create_branch(repo_path):
    run_cmd(["git", "fetch"], cwd=repo_path)
    branches = [b.strip().lstrip("* ").strip() for b in run_cmd(["git", "branch"], cwd=repo_path).splitlines()]
    if BRANCH_NAME not in branches:
        run_cmd(["git", "checkout", "-b", BRANCH_NAME], cwd=repo_path)
    else:
        run_cmd(["git", "checkout", BRANCH_NAME], cwd=repo_path)

def improve_code_with_ai(code):
    prompt = f"""
Mejora el siguiente código Python manteniendo su funcionalidad,
optimizando legibilidad y buenas prácticas de programación.
No incluyas explicaciones, solo devuelve el código final como bloque de código válido en Python.

Código:
{code}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw_output = response.choices[0].message.content

    # Extraer solo el contenido dentro del bloque ```python ... ```
    match = re.search(r"```python\s*(.*?)```", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Si no hay bloque de código, devolvemos todo
        return raw_output.strip()

def process_repo(repo_path):
    create_branch(repo_path)
    changed_files = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(tuple(CODE_EXTENSIONS)):
                file_path = Path(root) / file
                with file_path.open("r", encoding="utf-8") as f:
                    original_code = f.read()

                improved_code = improve_code_with_ai(original_code)
                if improved_code.strip() != original_code.strip():
                    with file_path.open("w", encoding="utf-8") as f:
                        f.write(improved_code)
                    changed_files.append(str(file_path.relative_to(repo_path)))

    if changed_files:
        run_cmd(["git", "add"] + changed_files, cwd=repo_path)
        run_cmd(["git", "commit", "-m", "Mejoras automáticas IA"], cwd=repo_path)
        run_cmd(["git", "push", "-u", "origin", BRANCH_NAME], cwd=repo_path)
    else:
        print(f"No hubo cambios en {repo_path}")

def main():
    for repo in REPOS_BASE_DIR.iterdir():
        if (repo / ".git").exists():
            print(f"\n--- Procesando repo: {repo.name} ---")
            try:
                process_repo(repo)
            except Exception as e:
                print(f"Error en {repo}: {e}")

if __name__ == "__main__":
    main()
