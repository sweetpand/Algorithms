class Solution:
    def shortestSuperstring(self, A: List[str]) -> str:
        n = len(A)
        common = {}
        for i, s1 in enumerate(A):
            for j in range(i):
                s2 = A[j]
                longest1, longest2 = 0, 0
                for k in range(min(len(s1), len(s2)) + 1):
                    if s1.endswith(s2[:k]):
                        longest1 = k
                    if s2.endswith(s1[: k]):
                        longest2 = k
                common[(i, j)] = longest1
                common[(j, i)] = longest2
        
        end = 1 << n
        dp = [[(float('inf'), '') 
               for i in range(n)] for m in range(end)]
        le, res = float('inf'), ''
        for used in range(end):
            for i, s in enumerate(A):
                if (used >> i) & 1 != 1:
                    continue
                mask = 1 << i
                if used ^ mask == 0:
                    dp[used][i] = (len(A[i]), A[i])
                else:
                    for j, last in enumerate(A):
                        if i == j or (used >> j) & 1 != 1:
                            continue
                        l, pref = dp[used ^ mask][j]
                        cand = pref + A[i][common[(j, i)] :]    
                        if len(cand) < dp[used][i][0]:
                            dp[used][i] = (len(cand), cand)
                if used == end - 1 and dp[used][i][0] < le:
                    le, res = dp[used][i]
        return res
