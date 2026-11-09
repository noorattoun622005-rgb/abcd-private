"""Convert notebooks/phase2_src.py (percent format) into notebooks/phase2.ipynb."""
import sys
from pathlib import Path
import nbformat

src = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/phase2_src.py").read_text()
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "notebooks/phase2.ipynb")
cells = []
for chunk in src.split("# %%")[1:]:
    if chunk.startswith(" [markdown]"):
        body = chunk.split("\n", 1)[1]
        lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in body.strip("\n").splitlines()]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.split("\n", 1)[1].strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                    "language_info": {"name": "python"}, "accelerator": "GPU"})
nbformat.write(nb, dst)
print(f"{dst}: {len(cells)} cells")
