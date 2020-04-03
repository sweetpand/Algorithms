"""
Given a linked list, determine if it has a cycle in it.
Follow up:
Can you solve it without using extra space?
Subscribe to see which companies asked this question.
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        walker = head
        runner = head.next
        try:
            while walker!=runner:
                walker = walker.next
                runner = runner.next.next
            return True
        except:
            return False
