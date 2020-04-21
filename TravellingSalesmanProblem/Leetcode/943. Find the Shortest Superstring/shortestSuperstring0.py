from functools import lru_cache
class Solution:
    def shortestSuperstring(self, A: List[str]) -> str:
        # TSP, DP 
        n = len(A)
        w = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(min(len(A[i]), len(A[j])), 0, -1):
                    if A[j].startswith(A[i][-k:]):
                        w[i][j] = k
                        break
        @lru_cache(None)
        def dp(nodes):
            if len(nodes) == 2:return A[nodes[0]]
            end = nodes[-1]
            end_idx = nodes.index(end) 
            # remove the current ending nodes from nodes[:-1]
            nodes_wo_end = nodes[:end_idx]+nodes[end_idx+1:-1]
            # A[end][w[node][end]:] will remove overlap part from the end node.
            return min((dp(nodes_wo_end +(node, ))+A[end][w[node][end]:]
                        for node in nodes_wo_end), key=len)
        return min((dp(tuple(range(n))+(node, )) for node in range(n)), key=len)
