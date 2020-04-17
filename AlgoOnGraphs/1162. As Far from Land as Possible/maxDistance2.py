class Solution(object):
    def maxDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        N, M = len(grid), len(grid[0]) if grid else 0
        def around(r,c, val):
            """return valid cells around (r, c) whose values are equal to val"""
            for (rr,cc) in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
                    if 0<=rr<N and 0<=cc<M and grid[rr][cc]==val:
                        yield (rr, cc)
        
        # mark the grid with SEEN values
        SEEN = 2
        
        # frontier. initially land-cells
        f = []
        for r in range(N):
            for c in range(M):
                if grid[r][c]:
                    f.append((r,c))
        # new frontier
        nf = []
        
        if not f:
            # no land cells
            return -1
        
        # distance away from land for the current frontier.
        dst = 0
        
        # BFS
        while f:
            while f:
                r, c = f.pop()
                for cell in around(r, c, 0):
                    rr, cc = cell
                    grid[rr][cc] = SEEN
                    nf.append(cell)
            f, nf = nf, f
            if f:
                dst += 1
        
        return dst or -1
