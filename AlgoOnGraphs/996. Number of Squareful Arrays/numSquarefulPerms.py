Explanation:

Count numbers ocuurrence.
For each number i, find all possible next number j that i + j is square.
Backtracking using dfs.
Time Complexity
It's O(N^N) if we have N different numbers and any pair sum is square.
We can easily make case for N = 3 like [51,70,30].

Seems that no hard cases for this problem and int this way it reduces to O(N^2).


def numSquarefulPerms(self, A):
        c = collections.Counter(A)
        cand = {i: {j for j in c if int((i + j)**0.5) ** 2 == i + j} for i in c}
        self.res = 0
        def dfs(x, left=len(A) - 1):
            c[x] -= 1
            if left == 0: self.res += 1
            for y in cand[x]:
                if c[y]: dfs(y, left - 1)
            c[x] += 1
        for x in c: dfs(x)
        return self.res
        
        
        
        
        
