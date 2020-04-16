
def compute():
	ans = sum(i for i in range(1000000) if is_decimal_binary_palindrome(i))
	return str(ans)


def is_decimal_binary_palindrome(n):
	s = str(n)
	if s != s[ : : -1]:
		return False
	t = bin(n)[2 : ]
	return t == t[ : : -1]


if __name__ == "__main__":
	print(compute())
