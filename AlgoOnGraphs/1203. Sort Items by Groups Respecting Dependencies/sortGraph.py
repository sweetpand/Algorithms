class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
	   # graphs initialization
        def makeGraphs():
            items = collections.defaultdict(dict)
            groups = {}
            for i, g in enumerate(group):
                items[g][i] = set()
                if g not in groups:
                    groups[g] = set()

            for i, ancestors in enumerate(beforeItems):
                for anc in ancestors:
                    if group[i] != group[anc]:
                        groups[group[anc]].add(group[i])
                    else:
                        items[group[i]][anc].add(i)
            return items, groups
        
        items, groups = makeGraphs()

        def sortGraph(G, stack):
            visiting = set() # a cycle is detected when we visit a node that is been visited
            visited = set()
            def dfs(node):
                if node in visiting: return False
                if node in visited: return True
                
                visiting.add(node)
                visited.add(node)
                for child in G[node]:
                    if not dfs(child):
                        return False
                stack.append(node)
                visiting.remove(node)
                return True
            
            for node in G:
                if not dfs(node):
                    return [] # this will indicate that there is a cycle inside the graph
            return stack
        
        groupsStack = []
        sortGraph(groups, groupsStack)
        if not groupsStack: # there is a cycle between some groups
            return []
        
        res = []
        for g in reversed(groupsStack):
            groupItems = []
            sortGraph(items[g], groupItems)
            if not groupItems: # cycle exists between items inside group g
                return []
            res.extend(reversed(groupItems))
        return res
