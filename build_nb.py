"""Tiny notebook builder: turns a `#%%md` / `#%%code` script into an executed .ipynb.

nbformat/nbclient are unavailable offline, so we assemble the JSON by hand and
run the code cells ourselves, capturing stdout and any matplotlib figures so the
notebook renders with outputs on GitHub.
"""
import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        s = line.strip()
        if s in ("#%%md", "#%%code"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("markdown" if s == "#%%md" else "code"), []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def src(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build(in_path, out_path, workdir):
    cells_in = parse(open(in_path, encoding="utf-8").read())
    ns = {"__name__": "__main__"}
    out_cells, n_exec, failures = [], 0, []

    for kind, text in cells_in:
        if kind == "markdown":
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": src(text)})
            continue
        n_exec += 1
        plt.close("all")
        buf = io.StringIO()
        err = None
        try:
            with redirect_stdout(buf):
                exec(compile(text, f"<cell {n_exec}>", "exec"), ns)
        except Exception:
            err = traceback.format_exc()
            failures.append((n_exec, err))
        outputs = []
        printed = buf.getvalue()
        if printed:
            outputs.append({"output_type": "stream", "name": "stdout", "text": src(printed)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight",
                        facecolor="white")
            outputs.append({
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                         "text/plain": [f"<Figure {num}>"]},
                "metadata": {},
            })
        plt.close("all")
        if err:
            outputs.append({"output_type": "stream", "name": "stderr", "text": src(err)})
        out_cells.append({
            "cell_type": "code", "execution_count": n_exec, "metadata": {},
            "outputs": outputs, "source": src(text),
        })

    nb = {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0],
                              "mimetype": "text/x-python",
                              "file_extension": ".py",
                              "pygments_lexer": "ipython3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {out_path}: {len(out_cells)} cells, {n_exec} executed, "
          f"{len(failures)} failed")
    for num, tb in failures:
        print(f"\n--- cell {num} failed ---\n{tb}")
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(build(sys.argv[1], sys.argv[2], "."))
