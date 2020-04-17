first build the graph. {word: set(other words in the input list)} . If the word is long, then check each words in the list with it (len(A)**2) ; if word is short, check if the other words is in its permuation(len(A0)**2) .

then for each word in the list, BFS its all neighbors until to the end.
return the groups


from collections import deque, defaultdict 

class Solution:
    def numSimilarGroups(self, A: List[str]) -> int:
        N, W = len(A), len(A[0])
        def cmpS(a, b):
            res = 0
            for c1, c2 in zip(a, b):
                res += (c1 != c2)
                if res > 2:
                    return False 
            return res == 2 
        g = defaultdict(set)
        if N < W*W: 
            for i in range(len(A)):
                for j in range(len(A)):
                    if A[i] in g[A[j]] or cmpS(A[i], A[j]): 
                        g[A[i]].add(A[j])
        else:
            buckets = collections.defaultdict(set)
            for i, word in enumerate(A):
                L = list(word)
                for j0, j1 in itertools.combinations(range(W), 2):
                    L[j0], L[j1] = L[j1], L[j0]
                    buckets["".join(L)].add(i)
                    L[j0], L[j1] = L[j1], L[j0]
                    
            for i1, word in enumerate(A):
                for i2 in buckets[word]:
                    g[A[i1]].add(A[i2])
            
        v = set() 
        res = 0 

        for i in A:
          if not i in v:
            v.add(i) 
            q = deque([i])
            res += 1  
            while q: 
              cur = q.popleft()
              for nb in g[cur]:
                if not nb in v:
                    v.add(nb) 
                    q.append(nb)  

        return res
