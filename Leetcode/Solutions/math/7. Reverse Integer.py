"""
Reverse digits of an integer.
Example1: x = 123, return 321
Example2: x = -123, return -321
click to show spoilers.
Note:
The input is assumed to be a 32-bit signed integer. Your function should return 0 when the reversed integer overflows.
"""

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        n = x if x > 0 else -x
        res = 0
        while n:
            res = res * 10 + n % 10
            n = n / 10
        if res > 0x7fffffff:
            return 0
        return res if x > 0 else -res
