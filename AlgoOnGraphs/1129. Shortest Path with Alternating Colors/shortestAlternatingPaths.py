class Solution:
    def shortestAlternatingPaths(self, n: int, red_edges: List[List[int]], blue_edges: List[List[int]]) -> List[int]:
        
        def build_edge(edges):
            lst = [set() for _ in range(n)]
            for p1, p2 in edges:
                lst[p1].add(p2)
            return lst
        
        edges = [build_edge(red_edges), build_edge(blue_edges)]
        
        dic = {
            (0, 0): 0,
            (0, 1): 0,
        }
        
        for i in range(1, n):
            for color in [0, 1]:
                dic[(i, color)] = float(inf)
        
        stack = [(0, 0), (0, 1)]
        
        while stack:
            i, color = stack.pop(0)
            for j in edges[color][i]:
                if dic[(j, 1 - color)] == float(inf):
                    dic[(j, 1 - color)] = dic[(i, color)] + 1
                    stack.append((j, 1 - color))
        
        output = [min(dic[(i, 0)], dic[(i, 1)]) for i in range(n)]

        return [-1 if d == float(inf) else d for d in output]
