 def canReach(self, A, i):
        if 0 <= i < len(A) and A[i] >= 0:
            A[i] = -A[i]
            return A[i] == 0 or self.canReach(A, i + A[i]) or self.canReach(A, i - A[i])
        return False
