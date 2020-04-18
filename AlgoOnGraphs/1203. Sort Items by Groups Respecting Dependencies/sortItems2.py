def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        groupgraph = {}
        itemgraph = [collections.defaultdict(list) for i in range(m)]
        for i in range(n):
            if group[i] == -1:
                group[i] = m+i  # assign groups to those that don't have em
            else:
                itemgraph[group[i]][i] = []  # init graph for each group
            groupgraph[group[i]] = []  # edge list for the group[i] in the group graph
        
        for i in range(n):
            for j in beforeItems[i]:
                if group[i] != group[j]:
                    groupgraph[group[j]].append(group[i])  # intra-group dependency
                if group[i] == group[j]:
                    itemgraph[group[i]][j].append(i)  # inter-group dependency
        
        def dfs(graph, start, visited):
            stack = [(start, 0)] # call stack
            path = set([start]) # cycle detection
            vis = set([start]) # avoid repetition
            order = [] # sort order
            while len(stack):
                u, i = stack.pop()
                while i < len(graph[u]):
                    v = graph[u][i]
                    if v in path:
                        return None
                    if v not in vis:
                        break
                    i += 1
                    
                if i >= len(graph[u]):
                    order.append(u)
                    path.remove(u)
                    continue
                    
                stack.append((u, i+1))
                stack.append((v, 0))
                vis.add(v)
                path.add(v)
                
            return order
                
        
        def topo(graph):
            visited = set()
            order = []
            for i in list(graph.keys()):
                if i not in visited:
                    o = dfs(graph, i, visited)
                    if o == None:
                        return None
                    for u in o:
                        if u not in visited:
                            order.append(u)
                    visited.update(o)
            
            return order[::-1]
        
        groupsort = topo(groupgraph)  # toposort the groups in dependency order
        if groupsort == None:
            return []
        
        res = []
        for g in groupsort:  # group topo-sorted order
            if g >= m:
                res.append(g-m)
                continue
            gsort = topo(itemgraph[g])  # toposort the group
            if gsort == None:
                return []
            res.extend(gsort)
        return res
