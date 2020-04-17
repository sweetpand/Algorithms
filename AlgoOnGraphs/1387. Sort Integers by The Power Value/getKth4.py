class Solution:
	def getKth(self, lo: int, hi: int, k: int) -> int:
		pow = []
		for i in range(lo, hi + 1):
			cnt = 0
			num = i
			while i != 1:
				if i % 2 == 0:
					i //= 2
				else:
					i = 3 * i + 1
				cnt += 1
			pow.append((num, cnt))
		pow.sort(key = lambda x: (x[1], x[0]))
		return pow[k - 1][0]
