import nbformat
from nbconvert import MarkdownExporter

# Load notebook
with open("files-directories.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# Convert to markdown
md_exporter = MarkdownExporter()
(body, resources) = md_exporter.from_notebook_node(nb)

# Save markdown
with open("files-directories.md", "w", encoding="utf-8") as f:
    f.write(body)

print("Converted notebook to Markdown!")
