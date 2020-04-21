class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        if any(len(g)==0 for g in graph):return 0
        N = len(graph)
        def bfs():
            from collections import deque
            Q = deque([(i, 0, {i}) for i in range(N)])
            pruning = collections.defaultdict(lambda : float('inf'))
            while Q:
                i, d, seen = Q.popleft()
                #pruning[tuple(sorted(seen))+(i,)]=d
                if len(seen)==N:return d
                for j in graph[i]:
                    this_seen = seen | {j}
                    this_key = tuple(sorted(this_seen))+(j,)
                    if pruning[this_key]>d+1:
                        pruning[this_key] = d+1
                        Q.append((j, d+1, this_seen))
        return bfs()
