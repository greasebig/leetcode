from typing import *
import numpy as np
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        x, y = len(nums1), len(nums2)
        low, high = 0, x
        while low <= high:
            partitionX = (low + high) // 2
            partitionY = (x + y + 1) // 2 - partitionX
            maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minX = float('inf') if partitionX == x else nums1[partitionX]

            maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minY = float('inf') if partitionY == y else nums2[partitionY]
            if maxX <= minY and maxY <= minX:
                if (x + y) % 2 == 0:
                    return (max(maxX, maxY) + min(minX, minY)) / 2
                else:
                    return max(maxX, maxY)
            elif maxX > minY:
                high = partitionX - 1
            else:
                low = partitionX + 1
solu=Solution()
nums1 =[1,2]
nums2 =[3,4]
print(solu.findMedianSortedArrays(nums1,nums2))


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        myreverse = None
        while head:
            nextnode = head.next  #定义第二个节
            head.next = myreverse   # 将首节点的next指向p，第一次执行时指向none,再次执行时指向断掉的部分节点
            myreverse = head  #将断掉的节点（即原来链表中前面的节点）赋给p
            head = nextnode #把头节点移向下一个节点，此时，这个节点也是一个不连续节点
        return myreverse

## 矩阵乘法
class Solution:
    def matrixmultiple(self, A, B):
        result = [[0 for _ in range(len(A))] for _ in range(len(B[0]))]  # len(A) A的行， len(B[0]) B的列
        for row in range(len(A)):
            for col in range(len(B[0])):
                for k in range(len(B)):
                    result[row][col] += A[row][k] * B[k][col]
        return result
solu=Solution()
nums1 =[[1,2],[3,4]]
nums2 =[[1,2],[3,4]]
print(solu.matrixmultiple(nums1,nums2))
nums1 =[1,2]
nums2 =[3,4]

# 1
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        l=len(nums)
        for i in range(l):
            for j in range(i+1,l):
                if nums[i]+nums[j]==target:
                    return [i,j]
solu=Solution()
print(solu.twoSum([3,2,4],6))

# 2
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        resutl_node=ListNode()
        residul=0
        l3=resutl_node
        while l1 or l2:
            l3.next=ListNode()
            l3=l3.next
            x=0 if l1==None else l1.val
            y=0 if l2==None else l2.val
            calcul=x+y+residul
            residul=calcul//10
            l3.val=calcul%10
            if l1:
                l1=l1.next
            if l2:
                l2=l2.next
        if residul!=0:
            l3.next=ListNode(val=residul)
        return resutl_node.next
solu=Solution()
l1_list= [2,4,3]
l1=ListNode()
next_node=l1
for i in range(len(l1_list)):
    next_node.next=ListNode()
    next_node=next_node.next
    next_node.val=l1_list[i]
l1=l1.next
l2_list= [5,6,4]
l2=ListNode()
next_node=l2
for i in range(len(l2_list)):
    next_node.next=ListNode()
    next_node=next_node.next
    next_node.val=l2_list[i]
l2=l2.next
print(solu.addTwoNumbers(l1,l2))

# 3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left_pointer=0
        max_length=0
        char_record_map={}
        for right_pointer in range(len(s)):
            if s[right_pointer] in char_record_map and right_pointer>=left_pointer:
                left_pointer=max(left_pointer,char_record_map[s[right_pointer]]+1)
            char_record_map[s[right_pointer]]=right_pointer
            max_length=max(max_length,right_pointer-left_pointer+1)
        return max_length
solu=Solution()
print(solu.lengthOfLongestSubstring("dvdf"))
print(solu.lengthOfLongestSubstring("pwwkew"))
