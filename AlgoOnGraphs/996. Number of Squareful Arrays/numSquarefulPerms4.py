class Solution(object):
    def numSquarefulPerms(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        l=len(A)
        g=collections.defaultdict(set)
        for i in range(l):
            for j in range(i+1, l):
                if (int((A[i]+A[j])**0.5))**2==A[i]+A[j]:
                    g[i].add(j)
                    g[j].add(i)
        
        # hamilton path dp[state][endIdx]
        dp=[[0]*l for _ in range(1<<l)]
        for i in range(l):
            dp[1<<i][i]=1
        
        # start from smaller state to larger state
        for i in range(1<<l):
            for j in range(l):
                # start from dp[i][j], check j in included in i
                if i&(1<<j):
                    for k in g[j]:
                        # add new Idx k to state i, check k is not included in i
                        if i&(1<<k)==0:
                            # new state is dp[i|(1<<k)][k]
                            dp[i|(1<<k)][k]+=dp[i][j]

        s=sum(dp[(1<<l)-1][i] for i in range(l))
        # remove duplicates, divide by factorial for each appearance
        c=collections.Counter(A)
        for v in c.values():
            for vv in range(1, v+1):
                s/=vv
        return s
