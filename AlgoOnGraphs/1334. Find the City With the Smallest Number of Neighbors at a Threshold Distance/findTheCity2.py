import heapq
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        cost = [[float('inf')]*n for _ in range(n)]
        graph = collections.defaultdict(list)
        for a,b,c in edges:
            cost[a][b]=cost[b][a]=c
            graph[a].append(b)
            graph[b].append(a)
            
        def dijkstra(i):
            dis = [float('inf')]*n
            dis[i], pq =0, [(0, i)]
            heapq.heapify(pq)
            while pq:
                d, i = heapq.heappop(pq)
                if d> dis[i]:continue
                for j in graph[i]:
                    this_cost = d+cost[i][j]
                    if this_cost<dis[j]:
                        dis[j] = this_cost
                        heapq.heappush(pq, (this_cost, j))
            return sum(d<=distanceThreshold for d in dis)-1
        res = {dijkstra(i):i for i in range(n)}
        return res[min(res)]
