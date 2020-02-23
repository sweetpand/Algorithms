A = [-2, 1, 2, 4, 7,11]
target = 13
#TC:O(n^2)
#SC: O(1)
def two_sum_brute_force(A, target):
	for i in range(len(A) - 1):
		for j in range(i + 1, len(A)):
			if A[i] + A[j] == target:
				print(A[i], A[j])
				return True 
	return False

#TC: O(n)
#SC: O(n)
def two_sum_hash_table(A, target):
	hasht = dict()
	for i in range(len(A)):
		if A[i] in hasht:
			print(hasht[A[i]], A[i])
			return True
		else:
			hasht[target - A[i]] = A[i]
	return False

#TO: O(n)
#SC: O(1)
def two_sum(A, target):
    i = 0
    j = len(A) - 1
    while i < j:
        if A[i] + A[j] == target:
            print(A[i], A[j])
            return True
        elif A[i] + A[j] < target:
            i += 1
        else:
            j -= 1
    return False

print(two_sum(A, target))
#print(two_sum_hash_table(A, target))
#print(two_sum_brute_force(A, target))