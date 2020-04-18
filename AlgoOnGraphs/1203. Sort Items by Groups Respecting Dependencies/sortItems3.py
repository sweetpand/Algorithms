class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        def dfs(node, Adj, ans):
            GREY.add(node)
            for nei in Adj[node]:
                if nei in GREY:
                    return False
                if nei in BLACK:
                    continue
                if not dfs(nei, Adj, ans):
                    return False
            GREY.remove(node)
            BLACK.add(node)
            ans.appendleft(node)
            return True
        
        Adj_item = dict()
        for node in range(n):
            Adj_item[node] = []
            
        for node in range(len(beforeItems)):
            for nei in beforeItems[node]:
                Adj_item[nei].append(node)
        
        GREY = set()
        BLACK = set()
        item_level_sort = collections.deque()
        for node in range(n):
            if node not in BLACK:
                if not dfs(node, Adj_item, item_level_sort):
                    return []
        
        group_id = m
        for node in range(len(group)):
            if group[node] == -1:
                group[node] = group_id
                group_id+=1
        Adj_group = dict()
        for node in range(group_id):
            Adj_group[node] = []
        
        for node in range(len(beforeItems)):
            for nei in beforeItems[node]:
                if group[node] != group[nei]:
                    Adj_group[group[nei]].append(group[node])
        
        GREY = set()
        BLACK = set()
        group_level_sort = collections.deque()
        for node in range(group_id):
            if node not in BLACK:
                if not dfs(node, Adj_group, group_level_sort):
                    return []
        
        buckets = [[] for _ in range(group_id)]
        for node in item_level_sort:
            buckets[group[node]].append(node)
        
        ans = []
        for group in group_level_sort:
            ans += buckets[group]
        return ans
