import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "crew_app.py",
    "src/tools.py",
    "src/tasks.py",
    "src/agents.py",
    "requirements.txt",
    ".gitignore"

]

for file in list_of_files:
    file_path = Path(file)
    if not file_path.exists():
        print(f"Creating {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    else:
        print(f"{file_path} already exists.")


