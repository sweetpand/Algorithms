
import datetime


# We simply use Python's built-in date library to compute the answer by brute force.
def compute():
	ans = sum(1
		for y in range(1901, 2001)
		for m in range(1, 13)
		if datetime.date(y, m, 1).weekday() == 6)
	return str(ans)


if __name__ == "__main__":
	print(compute())
