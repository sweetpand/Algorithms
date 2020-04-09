# 
# Solution to Project Euler problem 53

import eulerlib


def compute():
	ans = sum(1
		for n in range(1, 101)
		for k in range(0, n + 1)
		if eulerlib.binomial(n, k) > 1000000)
	return str(ans)


if __name__ == "__main__":
	print(compute())
