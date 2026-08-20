"""
Jane Street puzzle -- "'Pent-Up' Frustration 3 / Knight Moves 7" (July 2026)
============================================================================

Answer: 33609

Run:
    python solver.py            # solve + verify + pretty-print everything
    python solver.py --quiet    # just print the answer

The board (an 8x8 grid) is tiled by the twelve pentominoes plus one 2x2
tetromino, giving 13 regions.  Every region is made of 1x1x1 cubes, and each
region gets exactly one extra cube ("tower") on one of its squares.  So every
square has altitude 1, except the 13 tower squares which have altitude 2.

A move travels 0 units in one dimension, 1 in another and 2 in the third,
where the third dimension is altitude.  Because altitudes are only ever 1 or 2,
|dz| <= 1, so |dz| = 2 is impossible and only two move types survive:

    |dz| = 0  ->  {|dr|, |dc|} = {1, 2}   ordinary knight move, same altitude
    |dz| = 1  ->  {|dr|, |dc|} = {0, 2}   straight 2-square jump, altitude flips

Scoring: the knight starts at the bottom-left square with score 0.  On move N
the score is  +N  (level move),  *N  (moving up) or  /N  (moving down, allowed
only when the score is divisible by N).

The knight never repeats a square and stops the moment it has stood on all 13
towers.  Scores were written down at moves 0, 3, 6, 9, 12, 15, 18 and then
every K moves for some K > 3.  Twelve numbers are given, so there are twelve
recorded moves: 7 of them at 0..18, hence exactly 5 afterwards.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. The board                                                                #
# --------------------------------------------------------------------------- #

# Region layout, read straight off the puzzle image (see notebook for how the
# thick borders were detected programmatically).  Row 0 is the TOP row.
REGION_ROWS: Tuple[str, ...] = (
    "AAAAABBB",
    "CCCDDEEB",
    "CFCDDDEB",
    "GFFHHIEE",
    "GGFHHIIJ",
    "GKFLIIMJ",
    "GKLLLMMJ",
    "KKKLMMJJ",
)

# The twelve given scores, keyed by cell index (index = 8 * row + col).
GIVEN: Dict[int, int] = {
    0 * 8 + 5: 37,    0 * 8 + 7: 1100,
    2 * 8 + 3: 23,    2 * 8 + 5: 138,
    3 * 8 + 0: 528,
    4 * 8 + 1: 449,   4 * 8 + 4: 16,
    5 * 8 + 1: 750,   5 * 8 + 3: 88,
    5 * 8 + 5: 272,   5 * 8 + 6: 1,
    7 * 8 + 0: 0,     # the bottom-left starting square
}

START = 7 * 8 + 0                       # bottom-left square, a1
LETTERS = sorted(set("".join(REGION_ROWS)))
NREG = len(LETTERS)                     # 13

REGION: List[int] = [LETTERS.index(REGION_ROWS[i // 8][i % 8]) for i in range(64)]
REGION_CELLS: List[List[int]] = [[i for i in range(64) if REGION[i] == g]
                                 for g in range(NREG)]
FULL_MASK: List[int] = [sum(1 << i for i in REGION_CELLS[g]) for g in range(NREG)]
NUMBERED_MASK: int = sum(1 << i for i in GIVEN)

# --------------------------------------------------------------------------- #
# 2. Move table                                                               #
# --------------------------------------------------------------------------- #

LEVEL_DELTAS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
FLIP_DELTAS = [(0, 2), (0, -2), (2, 0), (-2, 0)]

# MOVES[i] = list of (destination, flips_altitude)
MOVES: List[List[Tuple[int, int]]] = [[] for _ in range(64)]
for _i in range(64):
    _r, _c = divmod(_i, 8)
    for _flip, _deltas in ((0, LEVEL_DELTAS), (1, FLIP_DELTAS)):
        for _dr, _dc in _deltas:
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < 8 and 0 <= _nc < 8:
                MOVES[_i].append((_nr * 8 + _nc, _flip))

# --------------------------------------------------------------------------- #
# 3. Search                                                                   #
# --------------------------------------------------------------------------- #

Solution = Tuple[List[int], List[int], int]      # (path, tower per region, K)


def record_moves(K: int) -> Tuple[set, int, int]:
    """Recorded move indices for a given K, plus the legal path-length window."""
    records = set(range(0, 19, 3)) | {18 + j * K for j in range(1, 6)}
    return records, 18 + 5 * K, 18 + 6 * K - 1


def solve(K: int) -> List[Solution]:
    """Exhaustively find every walk consistent with the puzzle, for this K.

    Tower positions are *not* enumerated up front (5**12 * 4 possibilities).
    Instead each move forces the altitude of the square being entered, which
    incrementally decides -- or forbids -- towers:

      * a level move keeps the altitude, so the new square is a tower iff the
        old one was;
      * a straight jump flips it: from altitude 1 you must land on a tower,
        from a tower you must land on altitude 1.

    So `tower[g]` (the tower of region g, or -1) and `excluded[g]` (a bitmask of
    squares in g proven to be plain height 1) grow as the walk is extended, and
    a region whose every square is excluded is an immediate dead end.
    """
    records, last_record, max_len = record_moves(K)
    solutions: List[Solution] = []
    path: List[int] = [START]

    def dfs(pos: int, mv: int, score: int, visited: int,
            tower: List[int], excluded: List[int], alt: int, n_towers: int) -> None:
        if n_towers == NREG:
            # All 13 towers have been stood on: the walk stops right here.
            if last_record <= mv <= max_len:
                solutions.append((path.copy(), tower.copy(), K))
            return
        if mv >= max_len:
            return

        nmv = mv + 1
        recording = nmv in records

        for nxt, flips in MOVES[pos]:
            if visited >> nxt & 1:
                continue
            # Squares carrying a printed number are visited exactly on the
            # moves whose score was written down -- and never otherwise.
            if recording:
                if nxt not in GIVEN:
                    continue
            elif NUMBERED_MASK >> nxt & 1:
                continue

            n_alt = (3 - alt) if flips else alt        # 1 <-> 2
            g = REGION[nxt]
            n_tower, n_excluded, n_cnt = tower, excluded, n_towers

            if n_alt == 2:                             # nxt must be g's tower
                if tower[g] != -1 or excluded[g] >> nxt & 1:
                    continue
                n_tower = tower.copy()
                n_tower[g] = nxt
                n_cnt = n_towers + 1
            else:                                      # nxt must be plain
                if tower[g] == nxt:
                    continue
                if not (excluded[g] >> nxt & 1):
                    n_excluded = excluded.copy()
                    n_excluded[g] = excluded[g] | 1 << nxt
                    if tower[g] == -1 and n_excluded[g] == FULL_MASK[g]:
                        continue                       # region has nowhere left
            # Score update
            if not flips:
                n_score = score + nmv
            elif n_alt == 2:
                n_score = score * nmv
            else:
                if score % nmv:
                    continue
                n_score = score // nmv
            if recording and n_score != GIVEN[nxt]:
                continue

            path.append(nxt)
            dfs(nxt, nmv, n_score, visited | 1 << nxt,
                n_tower, n_excluded, n_alt, n_cnt)
            path.pop()

    for start_alt in (1, 2):
        tower = [-1] * NREG
        excluded = [0] * NREG
        g = REGION[START]
        if start_alt == 2:
            tower[g] = START
            n_towers = 1
        else:
            excluded[g] = 1 << START
            n_towers = 0
        dfs(START, 0, 0, 1 << START, tower, excluded, start_alt, n_towers)

    return solutions


def solve_all(k_range: Sequence[int] = range(1, 13)) -> List[Solution]:
    """Search every plausible K.  K > 3 is required by the puzzle text; K >= 10
    is impossible anyway because 18 + 5*10 = 68 exceeds the 63 available moves.
    """
    out: List[Solution] = []
    for K in k_range:
        out.extend(solve(K))
    return out


# --------------------------------------------------------------------------- #
# 4. Replay / verification / answer                                           #
# --------------------------------------------------------------------------- #

def replay(path: Sequence[int], tower: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int], List[tuple]]:
    """Re-derive altitudes, scores and a move log from scratch, asserting rules."""
    altitude = {path[0]: 2 if tower[REGION[path[0]]] == path[0] else 1}
    score = {path[0]: 0}
    running, log = 0, []

    for n in range(1, len(path)):
        p, q = path[n - 1], path[n]
        dr, dc = abs(p // 8 - q // 8), abs(p % 8 - q % 8)
        a_q = 2 if tower[REGION[q]] == q else 1
        dz = abs(a_q - altitude[p])
        assert sorted((dr, dc, dz)) == [0, 1, 2], f"illegal move #{n}"
        if dz == 0:
            running += n
            kind = f"+{n}"
        elif a_q > altitude[p]:
            running *= n
            kind = f"x{n}"
        else:
            assert running % n == 0, f"non-integer division on move #{n}"
            running //= n
            kind = f"/{n}"
        altitude[q], score[q] = a_q, running
        log.append((n, p, q, kind, running))

    assert len(set(path)) == len(path), "a square was revisited"
    assert sorted(tower) == sorted(set(tower)) and -1 not in tower
    assert set(tower) <= set(path), "some tower was never visited"
    return altitude, score, log


def neighbour_sums(path: Sequence[int], score: Dict[int, int]) -> Tuple[int, List[tuple]]:
    """Answer = total of, over unvisited squares, the sum of adjacent path scores."""
    visited = set(path)
    rows = []
    total = 0
    for cell in range(64):
        if cell in visited:
            continue
        r, c = divmod(cell, 8)
        parts = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and nr * 8 + nc in visited:
                parts.append(score[nr * 8 + nc])
        rows.append((cell, sorted(parts), sum(parts)))
        total += sum(parts)
    return total, rows


def check_recording_schedule(path: Sequence[int], score: Dict[int, int], K: int) -> None:
    records, last_record, max_len = record_moves(K)
    length = len(path) - 1
    assert last_record <= length <= max_len
    written = {n for n in records if n <= length}
    assert len(written) == len(GIVEN) == 12
    for n in sorted(written):
        cell = path[n]
        assert cell in GIVEN and GIVEN[cell] == score[cell], f"move {n} mismatch"
    for n, cell in enumerate(path):
        assert (cell in GIVEN) == (n in written), f"move {n} on a printed square"


# --------------------------------------------------------------------------- #
# 5. Pretty printing                                                          #
# --------------------------------------------------------------------------- #

def cell_name(cell: int) -> str:
    """Chess-style name: column a-h left to right, row 1-8 bottom to top."""
    r, c = divmod(cell, 8)
    return f"{'abcdefgh'[c]}{8 - r}"


def grid(values: Dict[int, object], width: int = 6, blank: str = ".") -> str:
    return "\n".join(
        " ".join(str(values.get(r * 8 + c, blank)).rjust(width) for c in range(8))
        for r in range(8)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="print only the answer")
    args = ap.parse_args()

    sols = solve_all()
    assert len(sols) == 1, f"expected a unique solution, found {len(sols)}"
    path, tower, K = sols[0]
    altitude, score, log = replay(path, tower)
    check_recording_schedule(path, score, K)
    answer, rows = neighbour_sums(path, score)

    if args.quiet:
        print(answer)
        return 0

    print(f"unique solution: K = {K}, {len(path) - 1} moves, {len(path)} squares visited")
    print("towers:", ", ".join(f"{LETTERS[g]}{cell_name(tower[g])}" for g in range(NREG)))
    print("\npath:", " -> ".join(cell_name(c) for c in path))

    print("\nmove log")
    print(" #   from  to   op    score")
    for n, p, q, kind, val in log:
        print(f"{n:>2}   {cell_name(p):>4}  {cell_name(q):<4} {kind:<5} {val}")

    print("\ncompleted score grid (row 8 at top)")
    print(grid(score))
    print("\naltitudes (2 = tower)")
    print(grid(altitude, width=1))

    print("\nunvisited squares and their neighbour sums")
    for cell, parts, s in rows:
        print(f"  {cell_name(cell):>3}  {' + '.join(map(str, parts)) or '0':<28} = {s}")

    print(f"\nANSWER = {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
