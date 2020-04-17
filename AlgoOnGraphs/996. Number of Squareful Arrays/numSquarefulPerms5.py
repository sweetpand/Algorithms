backtracking solution, O(n!)

class Solution(object):
    def numSquarefulPerms(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        self.res=0
        counter=collections.Counter(A)
        g=collections.defaultdict(set)
        l=len(A)
        for i in range(l):
            for j in range(i+1, l):
                if (int((A[i]+A[j])**0.5))**2==A[i]+A[j]:
                    g[A[i]].add(A[j])
                    g[A[j]].add(A[i])
        
        def backtracking(val, idx=0):
            if idx==l-1:
                self.res+=1
                return
            counter[val]-=1
            for nei in g[val]:
                if counter[nei]:
                    backtracking(nei, idx+1)
            counter[val]+=1
            
        for v in counter.keys():
            backtracking(v)
        return self.res
