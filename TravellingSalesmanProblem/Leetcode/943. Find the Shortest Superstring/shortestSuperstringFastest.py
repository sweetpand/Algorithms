class Solution:
    def shortestSuperstring(self, words: List[str]) -> str:
        def concat(s, t, mink):
            for k in range(min(len(s), len(t)) - 1, 0, -1):
                if k <= mink: break
                if s[-k:] == t[:k]: return k, s + t[k:]
                if t[-k:] == s[:k]: return k, t + s[k:]
            return 0, s + t
        
        if not words: return ''
        while len(words) > 1:
            sharedsize = a = b = -1
            concatstr = ''
            for j in range(len(words)):
                for i in range(j):
                    k, s = concat(words[i], words[j], sharedsize)
                    if k > sharedsize:
                        sharedsize, concatstr = k, s
                        a, b = i, j
            if sharedsize > 0:
                words[b] = concatstr
                words[a] = words[-1]
            else:
                words[0] += words[-1]
            words.pop()
        return words[0]
