import os
from pathlib import Path

list_of_files = [
"README.md",
"research/note-1.ipynb",
"dataset/info.txt",
"src/__init__.py",
"src/model.py",
"src/train.py",
"src/data_prepare.py",
"src/evaluation.py",
"gans.py"
]

for file in list_of_files:
    file_path = Path(file)

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        print(f"Created : ", {file_path})
    else:
        print(f"File path does't exists")
