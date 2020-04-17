from functools import cmp_to_key
class Solution(object):
    def __init__(self):
        self.d = dict()
        self.d[1] = 0
    def getPower(self,num):
        if num in self.d:
            return self.d[num]
        if num % 2 == 0:
            self.d[num] = self.getPower(num//2) + 1
        else:
            self.d[num] = self.getPower(3*num +1)+1
        return self.d[num]
    
    def getKth(self, lo, hi, k):
        """
        :type lo: int
        :type hi: int
        :type k: int
        :rtype: int
        """
        def compare(x,y):
            powerX = self.getPower(x)
            powerY = self.getPower(y)
            if powerX == powerY:
                return x-y
            return powerX - powerY
        
        arr = list(range(lo,hi+1))
        arr.sort(key=cmp_to_key(compare))
        return arr[k-1]
