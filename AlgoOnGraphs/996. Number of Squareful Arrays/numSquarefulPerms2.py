since A.length <= 12, we can search A to find potential partner for each number in A
use DFS to do backtracking by going through each possibility


class Solution:
    def numSquarefulPerms(self, A: 'List[int]') -> 'int':
        self.unique = collections.Counter(A)
        self.partner = {x:{y for y in self.unique if (x + y) ** 0.5 == int((x + y) ** 0.5)} for x in self.unique}
        self.ans = 0
        for x in self.unique:
            self.dfs(x, len(A)-1)
        return self.ans
        
    def dfs(self, x: 'int', remain: 'int') -> 'Void':
        if remain == 0:
            self.ans += 1
            return
        
        self.unique[x] -= 1
        for y in self.partner[x]:
            if self.unique[y] > 0:
                self.dfs(y, remain - 1)
        self.unique[x] += 1
