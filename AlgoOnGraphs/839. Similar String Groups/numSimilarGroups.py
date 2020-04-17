Simiar idea as the solution. Using union find set. Just a bit verbose in implementation to ensure no redundandy on performance.


class UFS: 
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.ranks = [1 for _ in range(n)]
    def find(self, u):
        while self.parents[u] != u:
            self.parents[u] = self.parents[self.parents[u]]
            u = self.parents[u]
        return u
    def union(self, u, v):
        p1, p2 = self.find(u), self.find(v)
        if p1 == p2:
            return False
        if self.ranks[p1] == self.ranks[p2]:
            self.parents[p2] = p1
            self.ranks[p1] += 1
        elif self.ranks[p1] > self.ranks[p2]:
            self.parents[p2] = p1
        else:
            self.parents[p1] = p2
        return True
    
class Solution:
    def numSimilarGroups(self, A: List[str]) -> int: # O(k^3*n), O(n*k) || O(k*n^2), O(n*k)
        if not A or not A[0]:
            return 0
        A = list(set(A))
        n, k = len(A), len(A[0])
        hm = {s:i for s,i in zip(A, range(n))}
        ufs = UFS(n)
        if k**2 < n:
            for s in A:
                temp = list(s)
                for i in range(k):
                    for j in range(i+1, k):
                        if temp[i] != temp[j]:
                            temp[i], temp[j] = temp[j], temp[i]
                            nei = ''.join(temp)
                            if nei in hm:
                                ufs.union(hm[nei], hm[s])
                            temp[i], temp[j] = temp[j], temp[i]
        else:
            print(A)
            for i in range(n):
                for j in range(i+1, n):
                    if self.isSimilar(A[i], A[j], k):
                        ufs.union(hm[A[i]], hm[A[j]])
        return len(set((ufs.find(i) for i in range(n))))
    
    def isSimilar(self, str1, str2, k):
        diff1, diff2 = [], []
        for i in range(k):
            if str1[i] != str2[i]:
                if not diff1:
                    diff1 = [str1[i], str2[i]]
                elif not diff2:
                    diff2 = [str1[i], str2[i]]
                else:
                    return False
        if diff1 and diff2:
            return (diff1[0] == diff2[1] and diff1[1] == diff2[0])
        else: 
            return not diff1 and not diff2
