def canReach(self, arr: List[int], start: int) -> bool:
        q, seen = collections.deque([start]), {start}
        while q:
            cur = q.popleft()
            if arr[cur] == 0:
                return True
            for child in cur - arr[cur], cur + arr[cur]:
                if 0 <= child < len(arr) and child not in seen:
                    seen.add(child)
                    q.append(child) 
        return False
