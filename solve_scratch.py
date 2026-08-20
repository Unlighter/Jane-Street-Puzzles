"""Scratch solver for Jane Street 'Pent-Up' Frustration 3 / Knight Moves 7."""
import sys
from itertools import count

REGION_ROWS = [
    "AAAAABBB",
    "CCCDDEEB",
    "CFCDDDEB",
    "GFFHHIEE",
    "GGFHHIIJ",
    "GKFLIIMJ",
    "GKLLLMMJ",
    "KKKLMMJJ",
]
LETTERS = sorted(set("".join(REGION_ROWS)))
NREG = len(LETTERS)
region = [LETTERS.index(REGION_ROWS[i // 8][i % 8]) for i in range(64)]
region_cells = [[i for i in range(64) if region[i] == g] for g in range(NREG)]
full_mask = [sum(1 << i for i in region_cells[g]) for g in range(NREG)]

GIVEN = {
    5: 37, 7: 1100,          # row 0
    2 * 8 + 3: 23, 2 * 8 + 5: 138,
    3 * 8 + 0: 528,
    4 * 8 + 1: 449, 4 * 8 + 4: 16,
    5 * 8 + 1: 750, 5 * 8 + 3: 88, 5 * 8 + 5: 272, 5 * 8 + 6: 1,
    7 * 8 + 0: 0,
}
START = 7 * 8 + 0
NUMBERED_MASK = sum(1 << i for i in GIVEN)

# ---- move table -------------------------------------------------------
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
STRAIGHT = [(0, 2), (0, -2), (2, 0), (-2, 0)]
moves = [[] for _ in range(64)]
for i in range(64):
    r, c = divmod(i, 8)
    for dr, dc in KNIGHT:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            moves[i].append((nr * 8 + nc, 0))       # 0 = same altitude
    for dr, dc in STRAIGHT:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            moves[i].append((nr * 8 + nc, 1))       # 1 = altitude flips


def solve(K, score_cap=None, verbose=False):
    """Search for paths whose score is recorded at moves 0,3,..,18 then every K."""
    records = set(range(0, 19, 3))
    for j in range(1, 6):
        records.add(18 + j * K)
    last_record = 18 + 5 * K
    max_len = 18 + 6 * K - 1
    solutions = []

    def dfs(pos, mv, score, visited, tower, excl, alt, npath):
        # `tower[g]` = tower cell of region g or -1; `excl[g]` = bitmask of
        # cells in g that are proven to be height 1.
        if npath == NREG:                     # every tower has been stepped on
            if last_record <= mv <= max_len:
                solutions.append((list(path), list(tower), K))
            return                            # the walk stops here either way
        if mv >= max_len:
            return
        nmv = mv + 1
        is_rec = nmv in records
        for nxt, flip in moves[pos]:
            if visited >> nxt & 1:
                continue
            if is_rec:
                if nxt not in GIVEN:
                    continue
            elif NUMBERED_MASK >> nxt & 1:
                continue
            nalt = (3 - alt) if flip else alt          # 1<->2 on straight moves
            g = region[nxt]
            ntower, nexcl, nnpath = tower, excl, npath
            if nalt == 2:
                if tower[g] != -1 or (excl[g] >> nxt & 1):
                    continue
                ntower = tower[:]
                ntower[g] = nxt
                nnpath = npath + 1
            else:
                if tower[g] == nxt:
                    continue
                if not (excl[g] >> nxt & 1):
                    nexcl = excl[:]
                    nexcl[g] = excl[g] | 1 << nxt
                    if tower[g] == -1 and nexcl[g] == full_mask[g]:
                        continue              # region can no longer host a tower
            if flip:
                if nalt == 2:
                    nscore = score * nmv
                else:
                    if score % nmv:
                        continue
                    nscore = score // nmv
            else:
                nscore = score + nmv
            if is_rec and nscore != GIVEN[nxt]:
                continue
            if score_cap is not None and nscore > score_cap:
                continue
            path.append(nxt)
            dfs(nxt, nmv, nscore, visited | 1 << nxt, ntower, nexcl, nalt, nnpath)
            path.pop()

    for start_alt in (1, 2):
        tower = [-1] * NREG
        excl = [0] * NREG
        g = region[START]
        if start_alt == 2:
            tower[g] = START
            npath = 1
        else:
            excl[g] = 1 << START
            npath = 0
        path = [START]
        dfs(START, 0, 0, 1 << START, tower, excl, start_alt, npath)
    return solutions


if __name__ == "__main__":
    for K in range(4, 10):
        sols = solve(K, score_cap=10 ** 9)
        print("K =", K, "-> solutions:", len(sols))
        for p, tw, k in sols:
            print("   len", len(p) - 1, "path", p)
