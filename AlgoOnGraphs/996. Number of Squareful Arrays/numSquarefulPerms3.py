class Solution:
    def numSquarefulPerms(self, A):
        dfs = lambda path : 1 if len(path)==len(A) else sum(dfs(path+[v]) for v in ([j for j in set(A) if int((path[-1]+j)**0.5)**2==path[-1]+j] if len(path)>0 else set(A)) if A.count(v)>path.count(v))
        return dfs([])
