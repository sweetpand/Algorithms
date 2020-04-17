import functools
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        @functools.lru_cache(None)
        def power(x: int) -> int:
            if x == 1: return 0
            if x % 2 == 0: return power(x // 2) + 1
            return power(3 * x + 1) + 1

        A = list(range(lo, hi + 1))
        powers = sorted((power(num), i) for i, num in enumerate(A))
        return A[powers[k-1][1]]
