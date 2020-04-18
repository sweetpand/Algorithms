Traverse the array following the rules until reaching 0 or exhausting all numbers reachable. For this problem, either BFS or DFS will do.

Implementation (244ms, 100%):

class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        seen = set()
        stack = [start]
        while stack: 
            i = stack.pop()
            seen.add(i)
            if arr[i] == 0: return True
            for j in (i-arr[i], i+arr[i]):
                if 0 <= j < len(arr) and j not in seen: stack.append(j)
            
        return False 
Analysis:
Time complexity O(N)
Space complexity O(N)
