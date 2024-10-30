import ast
import json
import os
import sys


def get_python_files(repo_path: str) -> set[str]:
    """Get all .py and .ipynb files in the repository."""
    py_files = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".ipynb")):
                py_files.add(os.path.join(root, file))
    return py_files


def get_imports_from_python_file(file_path: str) -> set[str]:
    """Parse .py file to get imported modules."""
    imports = set()
    with open(file_path, "r", encoding="utf-8") as file:
        root = ast.parse(file.read(), filename=file_path)
        for node in ast.walk(root):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                assert module_name is not None
                imports.add(module_name.split(".")[0])  # Use only top-level module name
    return imports


def get_imports_from_notebook(file_path: str) -> set[str]:
    """Extract imports from Jupyter notebooks, skipping magic and shell commands."""
    imports = set()
    with open(file_path, "r", encoding="utf-8") as file:
        notebook = json.load(file)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                # Filter out lines starting with % or !
                code = "".join(line for line in cell.get("source", []) if not line.strip().startswith(("%", "!")))
                root = ast.parse(code)
                for node in ast.walk(root):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                        assert module_name is not None
                        imports.add(module_name.split(".")[0])
    return imports


def filter_stdlib_modules(modules: set[str]) -> set[str]:
    """Filter out standard library modules."""
    stdlib_modules = set(sys.stdlib_module_names)
    return modules - stdlib_modules


def list_local_modules(repo_path: str) -> set[str]:
    """List top-level local modules in the repository."""
    local_modules = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                relative_path = os.path.relpath(os.path.join(root, file), repo_path)
                module_name = relative_path.replace(os.sep, ".").rsplit(".py", 1)[0]
                local_modules.add(module_name.split(".")[0])
    return local_modules


def filter_local_modules(imports: set[str], repo_modules: set[str]) -> set[str]:
    """Filter out local modules from the imports set."""
    return imports - repo_modules


def main(repo_path: str):
    """Run the module extraction and filtering process."""
    py_files = get_python_files(repo_path)
    all_imports = set()

    for file_path in py_files:
        if file_path.endswith(".py"):
            all_imports.update(get_imports_from_python_file(file_path))
        elif file_path.endswith(".ipynb"):
            all_imports.update(get_imports_from_notebook(file_path))

    # Get local modules and filter them out along with standard library modules
    local_modules = list_local_modules(repo_path)
    external_modules = filter_local_modules(all_imports, local_modules)
    external_modules = filter_stdlib_modules(external_modules)

    print(f"Found {len(external_modules)} external packages:", sorted(external_modules))


# Usage: Replace 'your_repo_path' with the path to your repo
main(".")
