from collections import defaultdict
class Solution:

def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    def dfs(distance,heap,iset,x):
        while heap:
            weight,node = heapq.heappop(heap)
            if node in iset:
                continue
            iset.add(node)
            for j,w in graph[node]:
                if weight+w<distance[j] and distanceThreshold>= weight+w and j not in iset:
                    distance[j] = weight+w
                    heapq.heappush(heap,(weight+w,j))
        
    def util(i):
        visited.add(i)
        distance = {i:float('inf') for i in range(N)}
        distance[i] = 0
        heap = [(0,i)]
        dfs(distance,heap,set(),0)
        return distance
    
    graph = defaultdict(list)
    for u,v,w in edges:
        graph[u].append((v,w))
        graph[v].append((u,w))
        
    count = []
    N = n
    visited = set()
    res = []
    for i in range(N):
        if i not in visited:
            res.append(util(i))
			
    #count number of cities possible as per condition
    for i in res:
        x = 0
        for j in i:
            if 0<i[j]<float('inf'):
                x+=1
        count.append(x)
		
    #get the index
    min_v = float('inf')
    max_idx = float('-inf')
    for i in range(N):
        if count[i] <= min_v:
            min_v = count[i]
            if i>max_idx:
                max_idx = i
    return max_idx
