# Jane Street — “‘Pent-Up’ Frustration 3 / Knight Moves 7”

**Answer: `33609`**

Full worked solution to the [July 2026 Jane Street puzzle](https://www.janestreet.com/puzzles/pent-up-frustration-3-knight-moves-7-index/).

| file | what it is |
|---|---|
| [`solution.ipynb`](solution.ipynb) | the detailed write-up — derivation, code, figures, verification. Outputs are saved, so it renders fully in GitHub’s notebook preview without running anything |
| [`solver.py`](solver.py) | standalone script, no dependencies. `python solver.py` prints the walk, both grids and the answer |
| [`board.png`](board.png) | the puzzle image; the notebook reads the region tiling straight out of it |
| `nb_source.py`, `build_nb.py` | the `#%%md` / `#%%code` source of the notebook and the tiny builder that executes it into `.ipynb` |

## The result

* Regions: the 12 pentominoes plus one 2×2 tetromino, recovered from the image by measuring which grid lines are drawn thick.
* `K = 7`, so scores were recorded at moves **0, 3, 6, 9, 12, 15, 18, 25, 32, 39, 46, 53**.
* The walk is **54 moves / 55 squares**, starting on a1 (itself a tower) and ending on h6, the 13th tower:

```
a1 c2 e3 g3 h1 f2 e4 e6 e8 d6 c8 a7 a5 b7 d8 f8 d7 c5 d3 b2 a4 c4 e5 f7 h7 f6 d5 c3
a2 c1 e1 g1 f3 h3 h5 g7 f5 e7 c6 b4 a6 c7 b5 a3 b1 d2 b3 d4 e2 f4 g2 h4 g6 h8 h6
```

* Nine squares go unvisited — a8, b8, g8, b6, g5, g4, h2, d1, f1 — and their neighbour sums are
  44, 574, 1436, 2012, 1646, 1890, 9925, 8392, 7690 → **33609**.

## The idea that makes it easy

Every region is one layer of cubes plus exactly one extra cube, so altitudes are only
**1 or 2** and therefore `|Δz| ≤ 1`. A move whose three displacements are a permutation
of (0, 1, 2) then has just two possible forms:

* `|Δz| = 0` → an ordinary knight’s move, **same** altitude at both ends;
* `|Δz| = 1` → a straight two-square jump, altitude **flips**.

`|Δz| = 2` is impossible. So a straight jump from altitude 1 *must* land on a tower
(score `×N`) and a straight jump from a tower *must* land on a plain square (score `÷N`).

The consequence: the walk **determines** the tower layout instead of depending on it.
The search never enumerates the 5¹² × 4 ≈ 10⁹ possible layouts — it carries a partial
assignment (`tower[region]`, plus a mask of squares proven to be plain) that each move
either fixes or contradicts, and “has visited all the towers” is simply “all 13 regions
have an assigned tower”.

Two more constraints do the rest of the pruning: a printed number *is* the score on
arrival, so printed squares are entered **exactly** on recording moves and never on any
other; and `÷N` is only legal when `N` divides the score. The whole search (all
`K = 1..12`, no assumptions) visits ~11 million nodes in under 30 seconds and returns a
single walk.

## Running it

```bash
python solver.py            # ~18 s, prints everything
python solver.py --quiet    # just 33609
```

The notebook additionally needs `numpy`, `matplotlib` and `pillow`, because it
re-derives the board from `board.png`. To rebuild it after editing `nb_source.py`:

```bash
python build_nb.py nb_source.py solution.ipynb
```
