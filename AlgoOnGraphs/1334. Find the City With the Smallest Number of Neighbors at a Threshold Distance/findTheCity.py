Becasue O(N^3) is accepted in this problem, we don't need a very fast solution.
we can simply use Floyd algorithm to find the minium distance any two cities.

Iterate all point middle point k,
iterate all pairs (i,j).
If it go through the middle point k,
dis[i][j] = dis[i][k] + dis[k][j].


Complexity
Time O(N^3)
Space O(N^2) 

def findTheCity(self, n, edges, maxd):
        dis = [[float('inf')] * n for _ in xrange(n)]
        for i, j, w in edges:
            dis[i][j] = dis[j][i] = w
        for i in xrange(n):
            dis[i][i] = 0
        for k in xrange(n):
            for i in xrange(n):
                for j in xrange(n):
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
        res = {sum(d <= maxd for d in dis[i]): i for i in xrange(n)}
        return res[min(res)]
