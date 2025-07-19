import os
from pathlib import Path




list_of_files = [
    "README.md",
    "requirements.txt",
    "notebook/eperiments.ipynb",
    "assets/image-info.xlsx",
    "assets/dataset-info.txt",

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    """if filedir != "":
        os.mkdir(filedir, exist_ok=True)"""

    if filedir != "":
        try:
            os.mkdir(filedir)
        except FileExistsError:
            pass  # Directory already exists, ignore

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==  0):
        with open(filepath, "w") as f:
            pass # create an empty file
