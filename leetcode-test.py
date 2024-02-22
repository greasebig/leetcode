from typing import *
import numpy as np

## leetcode 4 寻找两个正序数组的中位数
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

## 链表反转
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

# 1.两数之和
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        l=len(nums)
        for i in range(l):
            for j in range(i+1,l):
                if nums[i]+nums[j]==target:
                    return [i,j]
solu=Solution()
print(solu.twoSum([3,2,4],6))

# 2.两数相加
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

# 3. 无重复字符的最长子串
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


# 4. 寻找两个正序数组的中位数
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:

        if len(nums1) > len(nums2):
            nums1,nums2=nums2,nums1  #？
        x = len(nums1)
        y = len(nums2)
        
        low,high=0,x
        while low <= high:
            partition_A = (low+high)//2  
            partition_B = (x+y+1)//2 - partition_A   #？

            X_little = float('-inf') if partition_A==0 else nums1[partition_A-1]  #？
            X_big = float('inf') if partition_A==x else nums1[partition_A]
            Y_little = float('-inf') if partition_B==0 else nums2[partition_B-1]
            Y_big = float('inf') if partition_B==y else nums2[partition_B]

            if X_little <= Y_big and Y_little <= X_big :
                if (x+y)%2 == 0:
                    return (max(X_little,Y_little) + min(X_big,Y_big))/2  #？
                else:
                    return max(X_little,Y_little)  #？
            elif Y_big < X_little:
                high = partition_A-1  #？
            else:
                low = partition_A+1  #？
#？边界，极端情况不知道如何设置


# 5. 最长回文子串
# 中心扩展法
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s)==0:
            return ""
        if len(s)==1:
            return s

        start = 0 
        end = 0

        for i in range(len(s)):
            len_self = self.expand_around_center(s,i,i)
            len_self_and_right = self.expand_around_center(s,i,i+1)
            max_len = max(len_self,len_self_and_right)

            if max_len > end - start:
                start = i - (max_len-1)//2
                end = i + max_len//2
        return s[start:end+1]

    def expand_around_center(self, s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
                len_it = right - left - 1
            return len_it
solu=Solution()
print(solu.longestPalindrome("babad"))

#动态规划
#s[i+1:j−1]是子问题，空间换时间，但是实际上时间空间都比中心扩散法差
def longestPalindrome(self, s: str) -> str:
        n = len(s)
        # 初始化一个二维数组，全部置为False
        dp = [[False] * n for _ in range(n)]

        start = 0  # 记录最长回文子串的起始位置
        max_len = 1  # 记录最长回文子串的长度

        # 所有长度为1的子串都是回文串
        for i in range(n):
            dp[i][i] = True

        # 遍历字符串，长度从2开始逐渐增加
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                # 判断当前子串是否是回文串
                if length == 2 and s[i] == s[j]:
                    dp[i][j] = True
                elif s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True

                # 更新最长回文子串的起始位置和长度
                if dp[i][j] and length > max_len:
                    start = i
                    max_len = length

        return s[start:start + max_len]

#自己写的动规
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]

        for i in range(n):
            dp[i][i] = True
        
        start = 0
        max_len = 1

        for length in range(2,n+1):  #！
            for i in range(n - length + 1): #！
                j = i + length -1 #！

                if length == 2 and s[i] == s[j]:
                    dp[i][j] = True
                elif s[i] == s[j] and dp[i+1][j-1] == True: #！
                    dp[i][j] = True
                
                if dp[i][j] == True and length > max_len:  #！
                    max_len = length
                    start = i

        return s[start:start+max_len]




# 6. Z 字形变换
# 找规律硬解，周期，边界多，时间空间也差，一个半小时，中等
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s  #！
        row = numRows
        length = len(s)
        column = (length//(numRows+numRows-2) + 1 ) * (numRows-1)  #！
        # 根据规律计算column大小，并额外添加一组
        dp = [["" for _ in range(column)] for _ in range(row)]  #！
        char_index = 0
        
        for current_column in range(column):
            for current_row in range(row):
                if current_column % (row - 1) == 0:  #！%和//总是分不清，用错
                    if char_index >= length:break  #！
                    dp[current_row][current_column] = s[char_index]  #！行列一开始写反
                    char_index += 1  #！

                elif current_row == row - current_column%(row-1) -1:  #！
                    if char_index >= length:break
                    dp[current_row][current_column] = s[char_index]
                    char_index += 1

        new_s = ''.join(dp[current_row][current_column] for current_row in range(row) for current_column in range(column))  #！
        return new_s
# 字符串相加只能使用 + 或者 ''.join()
# 不能使用.append(),append用在普通列表。虽然列表和字符串取值方式一样。


# 7. 整数反转
class Solution:
    def reverse(self, x: int) -> int:
        
        flag = 1
        if x<0:
            flag = -1
            x = -x
        new_num = 0
        while x != 0:
            a = x % 10
            new_num += a * 10 ** (len(str(x)) - 1)
            x = x // 10
        if new_num > 2 ** 31 - 1 or new_num < -2 ** 31: return 0
        return new_num if flag>0 else -new_num


# 9. 回文数
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 : return False

        new_num = 0
        new_num_list = []
        length = 0
        x2 = x  #！
        while x2 != 0:
            new_num_list.append(x2 % 10)
            x2 = x2 // 10
            length += 1
        for i in range(length):
            new_num += new_num_list[i] * 10 ** (length - i -1)
        return True if new_num == x else False  #！

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0) : return False
        new_num = 0
        new_num_list = []
        length = 0
        x2 = x
        while x2 > new_num:
            new_num = new_num * 10 + x2 % 10
            x2 = x2 // 10
        return new_num == x2 or new_num//10 == x2


# 10. 正则表达式匹配
# 标准答案dp
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]  # 尝试，去掉这*两个，看看是否匹配
                    if matches(i, j - 1):  # 如果*的上一个与字符匹配。参照后三行。
                        f[i][j] |= f[i - 1][j] # ？
                else:
                    if matches(i, j):  # 如果i,j匹配，则由上一组决定现在的dp
                        f[i][j] |= f[i - 1][j - 1] # 普通匹配
        return f[m][n]

# 自己重写dp
class Solution:
    def isMatch(self, s: str, p: str) -> bool: 
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]  #！
        dp[0][0] = True

        def match(i,j):
            if i == 0 : return False
            elif p[j-1] == '.' : return True
            elif s[i-1] == p[j-1] : return True     #！可以优化
            return False
        
        for i in range(m+1):  #！
            for j in range(1, n+1):  #！
                if p[j-1] == '*':
                    dp[i][j] |= dp[i][j-2]  #！
                    if match(i,j-1):
                        dp[i][j] |= dp[i-1][j]  #！
                elif match(i,j):
                    dp[i][j] |= dp[i-1][j-1]
        return dp[-1][-1]  #！


# 自己无法通过的代码，Solution().isMatch("aaa","ab*a*c*a")，硬匹配
class Solution:
    def isMatch(self, s: str, p: str) -> bool: 
        if len(p) > 2 :    
            if s[0] != p[0] and p[0] != '.' and p[1] != '*' : return False
        
        
        i = 0 # move index
        judge_num = i
        while s:
            if i >= len(p) : return False
            if p[i] == '*' :
                if p[i-1] == '.': 
                    s = s[1:]
                    judge_num += 1
                    continue
                if s[0] != p[i-1] : return False

            elif i + 1 < len(p):
                if s[0] != p[i] and p[i] != '.' and p[i+1] != '*' : 
                    return False
                elif s[0] != p[i] and p[i] != '.' and p[i+1] == '*' :
                    if i+2 < len(p) :
                        ii = i
                        while ii+2 < len(p) :
                            if p[i+2] == '*' : 
                                p = p[:i+1] + p[i+2:]
                                ii += 1
                            else : break
                        if p[i+2] == s[0] or p[i+2] == '.' : 
                            p = p[:i] + p[i+2:]
                            
                            
                    else: return False
                    
            elif i + 1 >= len(p):
                if s[0] != p[i] and p[i] != '.' : return False
            i += 1
            judge_num += 1
            s = s[1:]
        return True if len(p) == judge_num else False


# 11. 盛最多水的容器
# 双指针
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        length = len(height)
        left = 0
        right = length -1
        while left < right :
            area = (right - left) * min(height[right], height[left])
            if height[right] > height[left] : left += 1
            else : right -= 1
            if area > max_area : max_area = area
        return max_area

# 自己的，超出时间限制
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        length = len(height)
        for i in range(length):
            for j in range(i + 1, length):
                area = (j - i) * min(height[i], height[j])
                if area > max_area : max_area = area
        return max_area

