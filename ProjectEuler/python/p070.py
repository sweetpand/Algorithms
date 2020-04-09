# 
# Solution to Project Euler problem 70

import eulerlib


def compute():
	totients = eulerlib.list_totients(10**7 - 1)
	minnumer = 1
	mindenom = 0
	for (i, tot) in enumerate(totients[2 : ], 2):
		if i * mindenom < minnumer * tot and sorted(str(i)) == sorted(str(tot)):
			minnumer = i
			mindenom = totients[i]
	return str(minnumer)


if __name__ == "__main__":
	print(compute())
