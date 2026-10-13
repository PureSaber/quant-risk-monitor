"""Execute and save the audit notebook against the frozen forecast records."""

import json
import os

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from public_sources import ROOT


def main():
    run = ROOT / json.loads((ROOT / "active-run.json").read_text(encoding="utf8"))["run"]
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "jupyter-runtime")
    os.environ["IPYTHONDIR"] = str(ROOT / "ipython-runtime")
    nb = nbformat.read(run / "research-audit.ipynb", as_version=4)
    nbformat.validate(nb)
    NotebookClient(nb, timeout=120, startup_timeout=30, kernel_name="python3").execute(cwd=str(run))
    cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    nbformat.write(nb, run / "research-audit.ipynb")
    html, _ = HTMLExporter().from_notebook_node(nb)
    (run / "research-audit.html").write_text(html, encoding="utf8")
    result = {"executed_code_cells": len(cells), "errors": 0, "html_exported": True}
    (run / "notebook-execution.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result, flush=True)


if __name__ == "__main__":
    main()
