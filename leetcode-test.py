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

## 链表反转
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        reverse = None
        while head:
            nextnode = head.next  #定义第二个节
            head.next = reverse   # 将首节点的next指向p，第一次执行时指向none,再次执行时指向断掉的部分节点
            reverse = head  #将断掉的节点（即原来链表中前面的节点）赋给p
            head = nextnode #把头节点移向下一个节点，此时，这个节点也是一个不连续节点
        return reverse
# reverse none
# 1     - 2      - 3
# head
#      next_node
# reverse
#      head

    #      next_node
    # reverse



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



# 
class Solution:
    def myAtoi(self, s: str) -> int:
        number_list = []
        is_bigger_than_0 = True              #！
        has_symbol = stop = False              #！
        table = ['0','1','2','3','4','5','6','7','8','9']
        for index in range(len(s)) :
            if has_symbol and s[index] not in table : break
            if s[index] == ' ':
                if stop :
                    break
                continue
            elif s[index] == '.' :
                break
            elif not has_symbol and (s[index] == '-' or s[index] == '+') :              #！
                if stop :
                    break
                is_bigger_than_0 = False if s[index] == '-' else True
                has_symbol = True
                continue
            elif has_symbol and (s[index] == '-' or s[index] == '+') :              #！
                break
                
            elif s[index] in table :
                stop = True
                number_list.append(s[index])
            
            elif not stop : break              #！
            elif stop and s[index] not in table : break

            

        if not number_list : return 0              #！
        number_str = ''.join(number_list)
        number = int(number_str)              #！
        if is_bigger_than_0 :
            return 2 ** 31 - 1 if number > 2 ** 31 - 1 else number
        else : return -2 ** 31 if -number < -2 ** 31 else -number


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

# 自己重写dp 222
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



#  228 回头重写，方法忘记，记起又写不对
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * n] * m               #！
        dp[0][0] = True
        for i in range(1, m) :
            for j in range(1, n) :
                
                if s[i] == p[j] or p[j] == '.' :
                    dp[i][j] |= dp[i - 1][j - 1]
                if p[j] == '*' :
                    dp[i][j] |= dp[i][j - 2]
                    if s[i] == p[j - 1] or p[j - 1] == '.' :
                        dp[i][j - 1] |= dp[i - 1][j]              #！
        return dp[-1][-1]

# 228 更正
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1) ]             #！
        dp[0][0] = True

        def matches(i, j) : 
            if i == 0 : return False             #！
            elif s[i - 1] == p[j - 1] or p[j - 1] == '.' : return True

        for i in range(m + 1) :             #！
            for j in range(1, n + 1) :
                if p[j - 1] == '*' :
                    dp[i][j] |= dp[i][j - 2]
                    if matches(i, j - 1) :             #！
                        dp[i][j] |= dp[i - 1][j]             #！

                elif matches(i, j) :
                    dp[i][j] |= dp[i - 1][j - 1]
                
        return dp[-1][-1]





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



# 12. 整数转罗马数字
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        number = str(num)
        length = len(number)

        def len1(num1, index):
            if index == 1 : special1, special2, special3 = 'I', 'V', 'X'
            elif index == 2 : special1, special2, special3 = 'X', 'L', 'C'
            elif index == 3 : special1, special2, special3 = 'C', 'D', 'M'
            elif index == 4 : special1, special2, special3 = 'M', 'M', 'M'

            if num1 < 4 : return special1 * num1
            elif num1 == 4 : return special1 + special2
            elif num1 == 5 : return special2
            elif num1 > 5 and num1 < 9 :  return special2 + special1 * (num1 - 5)
            elif num1 == 9 : return special1 + special3

        roman = ''
        index = 1
        while num :
            roman = len1(num % 10, index) + roman
            num = num // 10
            index += 1
        return roman

# 13. 罗马数字转整数
# 一开始按照12思路不会写。换成hashmap以及一次读一个，且多看后一位
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        hashmap = {
                    'I': 1,
                    'V': 5,
                    'X': 10,
                    'L': 50,
                    'C': 100,
                    'D': 500,
                    'M': 1000,
                    }
        
        number = 0
        for i in range(len(s)) :
            if i + 1 < len(s) :
                if hashmap[s[i]] >= hashmap[s[i + 1]] :    #! 读hashmap用[] 而不是()
                    number += hashmap[s[i]]
                else :
                    number -= hashmap[s[i]]
            else :
                number += hashmap[s[i]]

        return number


# 14. 最长公共前缀
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        common_prefix = ''
        if len(strs) == 0 : return ''
        if len(strs) == 1 : return strs[0]
        min_length = float('inf')               # ！
        for word in strs :
            if len(word) < min_length : min_length = len(word)

        for char_index in range(min_length):
            for word in strs[1:] : 
                if word[char_index] != strs[0][char_index] : return common_prefix
                else : continue
            common_prefix += strs[0][char_index] 
        return common_prefix



# 15. 三数之和
# 排序 + 双指针
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        threesum = []
        threesum_dict = []

        import random
        def half_rank(numbers) :
            if len(numbers) < 2 : return numbers
            left_nums = []
            right_nums = []
            mid_nums = []
            pivot = random.randint(0, len(numbers) - 1)
            for element in  numbers:
                if element < numbers[pivot] : left_nums.append(element)
                elif element > numbers[pivot] : right_nums.append(element)
                else : mid_nums.append(element)
            return half_rank(left_nums) + mid_nums + half_rank(right_nums)

        rank_num = half_rank(nums)
        # nums.sort()
        # rank_num = nums   # 这个速度和空间都和half_rank一样
        
        for i in  range(len(rank_num) - 2) :
            if rank_num[i] > 0 : break            # ！
            if(i>0 and rank_num[i]==rank_num[i-1]):       # ！这两句判断加快，从308/313到311/313
                continue
            left = i + 1 
            right = len(rank_num) - 1
            while left < right :
                if rank_num[i] + rank_num[left] + rank_num[right] == 0 : 
                    
                    threesum.append([rank_num[i], rank_num[left], rank_num[right]])       # ！通过上面加的if，去掉dict查看重复值的步骤，最后通过
                    
                    while 1:
                        left += 1
                        if left < right :
                            if rank_num[left] != rank_num[left - 1] : break
                        else : break
                    while 1:    
                        right -= 1
                        if left < right :
                            if rank_num[right] != rank_num[right + 1] : break
                        else : break
                    
                elif rank_num[i] + rank_num[left] + rank_num[right] < 0 : 
                    while 1:
                        left += 1
                        if left < right :
                            if rank_num[left] != rank_num[left - 1] : break
                        else : break
                elif rank_num[i] + rank_num[left] + rank_num[right] > 0 : 
                    while 1:
                        right -= 1
                        if left < right :
                            if rank_num[right] != rank_num[right + 1] : break
                        else : break

        return threesum


# 以前是用 set() 再用 list() 做
# 现在是 超时 或者 重复 或者 漏
# 排序 + 双指针。 自己写。重复的不再判断，跳过。超时
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        threesum = []
        threesum_dict = []

        import random
        def half_rank(numbers) :
            if len(numbers) < 2 : return numbers
            left_nums = []
            right_nums = []
            mid_nums = []
            pivot = random.randint(0, len(numbers) - 1)
            for element in  numbers:
                if element < numbers[pivot] : left_nums.append(element)
                elif element > numbers[pivot] : right_nums.append(element)
                else : mid_nums.append(element)
            return half_rank(left_nums) + mid_nums + half_rank(right_nums)

        rank_num = half_rank(nums)
        
        for i in  range(len(rank_num) - 2) :
            left = i + 1 
            right = len(rank_num) - 1
            while left < right :
                if rank_num[i] + rank_num[left] + rank_num[right] == 0 : 
                    if {rank_num[i], rank_num[left], rank_num[right]} not in threesum_dict :
                        threesum.append([rank_num[i], rank_num[left], rank_num[right]])
                        threesum_dict.append({rank_num[i],rank_num[left], rank_num[right]})
                    else : 
                        while 1:
                            left += 1
                            if left < right :
                                if rank_num[left] != rank_num[left - 1] : break
                            else : break
                elif rank_num[i] + rank_num[left] + rank_num[right] < 0 : 
                    while 1:
                        left += 1
                        if left < right :
                            if rank_num[left] != rank_num[left - 1] : break
                        else : break
                elif rank_num[i] + rank_num[left] + rank_num[right] > 0 : 
                    while 1:
                        right -= 1
                        if left < right :
                            if rank_num[right] != rank_num[right + 1] : break
                        else : break

        return threesum

# 排序 + 双指针 。 自己写。超出时间限制
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        threesum = []
        threesum_dict = []

        def half_rank(numbers) :
            if len(numbers) < 2 : return numbers
            left_nums = []
            right_nums = []
            mid_nums = []
            pivot = len(numbers) // 2
            for element in  numbers:
                if element < numbers[pivot] : left_nums.append(element)
                elif element > numbers[pivot] : right_nums.append(element)
                else : mid_nums.append(element)
            return half_rank(left_nums) + mid_nums + half_rank(right_nums)

        rank_num = half_rank(nums)
        
        for i in  range(len(rank_num) - 2) :
            left = i + 1 
            right = len(rank_num) - 1
            while left < right :
                if rank_num[i] + rank_num[left] + rank_num[right] == 0 : 
                    if {rank_num[i], rank_num[left], rank_num[right]} not in threesum_dict :
                        threesum.append([rank_num[i], rank_num[left], rank_num[right]])
                        threesum_dict.append({rank_num[i],rank_num[left], rank_num[right]})
                    else : left += 1
                elif rank_num[i] + rank_num[left] + rank_num[right] < 0 : 
                    left += 1
                elif rank_num[i] + rank_num[left] + rank_num[right] > 0 : 
                    right -= 1

        return threesum

# 超时
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        length = len(nums)
        threesum = []
        threesum_norank = []
        for i in range(length) :
            for j in range(i+1, length) :
                
                for k in range(j+1, length) :
                    if nums[i] + nums[k] + nums[j] == 0 : 
                        if {nums[i], nums[j], nums[k]} not in threesum_norank :
                            threesum.append([nums[i], nums[j], nums[k]])
                            threesum_norank.append({nums[i], nums[j], nums[k]})
        return threesum







# 16. 最接近的三数之和
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        min_distance = float('inf')
        closest = 0
        for i in  range(len(nums) - 2) :
            if i > 0 and nums[i] == nums[i - 1] : continue
            left = i + 1 
            right = len(nums) - 1
            
            while left < right :
                summary = nums[i] + nums[left] + nums[right] 
                distance = summary - target
                if abs(distance) < min_distance :
                    min_distance = abs(distance)
                    closest = summary
                
                if distance == 0 : break
                elif distance < 0 :      # ！
                    
                    while left < right and nums[left] == nums[left + 1] :
                        left += 1
                    left += 1          # ！
                elif distance > 0 : 
                    
                    while left < right and nums[right] == nums[right - 1] :
                        right -= 1
                    right -= 1


        return closest



# 17. 电话号码的字母组合
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if digits == '' : return []
        table = {
            '2' : ['a', 'b', 'c'],
            '3' : ['d', 'e', 'f'],
            '4' : ['i', 'g', 'h'],
            '5' : ['j', 'k', 'l'],
            '6' : ['m', 'n', 'o'],
            '7' : ['p', 'q', 'r', 's'],
            '8' : ['t', 'u', 'v'],
            '9' : ['w', 'x', 'y', 'z'],    # ！
        }
        combile = []
        letter_group = []
        while digits :
            letter_group = letter_group + [table[digits[0]]]     # ！少了[]   列表就是concat
            digits = digits[1:]
        combile = letter_group[0] 
        for group in letter_group[1:] :
            new_combine = []
            for char1 in  combile :
                for char2 in group :
                    new_combine.append(char1 + char2)       # ！
            combile = new_combine
        return combile


# 18. 四数之和

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        foursum = []

        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1] : continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1] : continue        # ！ 关键在 j > i + 1   之前是 j > 1
                left = j + 1
                right = len(nums) - 1
                while left < right :
                    summary = nums[i] + nums[j] + nums[left] + nums[right]
                    if summary == target :
                        foursum.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1] :
                            left += 1
                        while left < right and nums[right - 1] == nums[right] :
                            right -= 1
                        left += 1
                        right -= 1
                    elif summary < target :
                        while left < right and nums[left] == nums[left + 1] :
                            left += 1
                        left += 1
                    elif summary > target :
                        while left < right and nums[right - 1] == nums[right] :
                            right -= 1
                        right -= 1
                
        return foursum


# 19. 删除链表的倒数第 N 个结点
# 双指针，快慢，测试用例中速度存储一样
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        fast = head
        dummy = ListNode(0, head)
        slow = dummy
        for _ in range(n) :
            fast = fast.next
        while fast :
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next


# 自己的反转两次
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        reverse = None
        while head :    # ！ 反转链表老是忘记
            next_node = head.next
            head.next = reverse
            reverse = head
            head = next_node

        reverse_again_and_delete = None
        head = reverse
        i = 0
        while head :
            i += 1
            if i == n :
                head = head.next
            else :
                next_node = head.next
                head.next = reverse_again_and_delete
                reverse_again_and_delete = head
                head = next_node
        return reverse_again_and_delete





# 20. 有效的括号
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) == 0 : return True
        elif len(s) == 1 : return False
        #left_symbol = ['(', '[','{']
        table = {
            '(' : ')',
            '[' : ']',
            '{' : '}',
        }
        left_symbol = table.keys()      # ！ 相比list用时从44ms提升到33ms
        left_list = []
        
        while s :
            if s[0] in left_symbol : 
                left_list.append(s[0])
                s = s[1:]
            else :
                if not left_list and s[0] : return False      # ！
                if table[left_list.pop()] == s[0] :      # ！
                    s = s[1:]
                else :
                    return False
        return False if left_list else True        # ！



# 21. 合并两个有序链表
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list2 : return list1
        elif not list1 : return list2
        
        if list1.val > list2.val : list1, list2 = list2, list1       # ！

        head1 = list1
        head2 = list2
        
        while head1 : 
            if head2.next :       # ！
                if head1.val <= head2.val and head1.next :
                    if head2.val <= head1.next.val :
                        present_node2 = head2       # ！
                        head2 = head2.next
                        next_node1 = head1.next       # ！
                        head1.next =  present_node2
                        head1 = head1.next
                        head1.next = next_node1    
                    else :
                        head1 = head1.next
                elif head1.val <= head2.val and not head1.next :
                    head1.next =  head2
                    break
                    
            elif not head2.next: 
                if head1.val <= head2.val and head1.next :
                    if head2.val <= head1.next.val :
                        present_node2 = head2
                        next_node1 = head1.next
                        head1.next =  present_node2
                        head1 = head1.next
                        head1.next = next_node1   
                        break        # ！
                    else :
                        head1 = head1.next
                elif head1.val <= head2.val and not head1.next :
                    head1.next =  head2
                    break
                
        return list1

l1_list= [5]
l1=ListNode()
next_node=l1
for i in range(len(l1_list)):
    next_node.next=ListNode()
    next_node=next_node.next
    next_node.val=l1_list[i]
l1=l1.next

l2_list= [1,2,4]
l2=ListNode()
next_node=l2
for i in range(len(l2_list)):
    next_node.next=ListNode()
    next_node=next_node.next
    next_node.val=l2_list[i]
l2=l2.next

print(Solution().mergeTwoLists(l1, l2))


# 22. 括号生成
# 想不出一镜到底的方法
# dfs
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        parenthesis = []
        
        def dfs(strs, left, right) :       # ！
            if left == 0 and right == 0 :
                parenthesis.append(strs)
                return
            if right < left :
                return
            if left > 0 :
                dfs(strs + '(', left - 1 ,right)
            if right > 0 :
                dfs(strs + ')', left ,right - 1)
        dfs('', n , n)
        return parenthesis


# 23. 合并 K 个升序链表

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists) == 0 : return None
        elif len(lists) == 1 : return lists[0]

        list1 = None
        index = 0
        for number in range(0, len(lists)) :   # ！上个版本使用while循环，然后自己对index加法，导致容易漏或越界
            index = number
            if lists[index] :
                list1 = lists[index]
                break

        if index == len(lists) - 1: return list1  # ！
        elif index == len(lists): return None
        
        for index2 in range(index + 1, len(lists)) :  # ！
            list2 = lists[index2]
            if not list2 : continue
            if list1.val > list2.val : list1, list2 = list2, list1
            head1 = list1
            head2 = list2
            while head1 :
                if head1.next :
                    if head2.next :
                        if head1.val <= head2.val and head2.val <= head1.next.val :
                            present_node2 = head2
                            head2 = head2.next
                            next_node1 = head1.next
                            head1.next = present_node2
                            present_node2.next = next_node1
                        else :
                            head1 = head1.next

                    else :
                        if head1.val <= head2.val and head2.val <= head1.next.val :
                            present_node2 = head2
                            next_node1 = head1.next
                            head1.next = present_node2
                            present_node2.next = next_node1
                            break
                        else :
                            head1 = head1.next

                else : 
                    head1.next = head2
                    break
        return list1


def makeListnode(l1_list) : 
    L1 = ListNode()
    next_node = L1
    for i in range(len(l1_list)):
        next_node.next=ListNode()
        next_node=next_node.next
        next_node.val=l1_list[i]
    return L1.next

l1 = makeListnode([1,4,5])
l2 = makeListnode([1])
l3 = makeListnode([2,6])
print(Solution().mergeKLists([None, l2]))


# 24. 两两交换链表中的节点
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head : return None
        dummy = ListNode(0, head)
        dummy_move = dummy  # ！
        while head :
            if not head.next : return dummy.next
            else :
                present_node = head
                next_node = head.next
                next_2_node = head.next.next

                present_node.next = next_2_node
                next_node.next = present_node  # ！
                
                dummy_move.next = next_node  # ！缺少这个，卡了好久
                dummy_move = present_node  # ！

                head = next_2_node

        return dummy.next

# 25. K 个一组翻转链表
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:

        if not head or k == 1 : return head       # ！
        dummy = ListNode(0, head)

        reverse_all = ListNode()
        reverse_k_group = reverse_all

        reverse = None
        number = 0
        
        while head :
            number += 1
            if number % k == 1 :       # ！
                first_node = head       # ！
                preserve_node = copy.deepcopy(head)       # ！
            present_node = head
            head = head.next
            
            present_node.next = reverse
            reverse = present_node
            
            
            if number % k == 0 : 
                reverse_k_group.next = present_node       # ！
                reverse_k_group = first_node       # ！
                reverse = None

        if number % k != 0 :
            reverse_k_group.next = preserve_node         # ！
        return reverse_all.next


# 官方，没有deepcopy，没有number,，速度快十倍，存储少一半
class Solution:
    # 翻转一个子链表，并且返回新的头与尾
    def reverse(self, head: ListNode, tail: ListNode):
        prev = tail.next
        p = head
        while prev != tail:
            nex = p.next
            p.next = prev
            prev = p
            p = nex
        return tail, head

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        hair = ListNode(0)
        hair.next = head
        pre = hair

        while head:
            tail = pre
            # 查看剩余部分长度是否大于等于 k
            for i in range(k):
                tail = tail.next
                if not tail:
                    return hair.next
            nex = tail.next
            head, tail = self.reverse(head, tail)
            # 把子链表重新接回原链表
            pre.next = head
            tail.next = nex
            pre = tail
            head = tail.next
        
        return hair.next



# 26. 删除有序数组中的重复项
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        left = 0        # ！想不到双指针，想用pop 或者新建列表，非原地， 或者set
        right = left + 1
        while left < right and right < len(nums) :
                
            if nums[left] != nums[right] :
                
                nums[left + 1] = nums[right]        # ！想用number记录，但是因为用了 == 导致错误
                left += 1
                right += 1

            else : 
                right += 1
        


        return left + 1 if nums else 0


# 27. 移除元素

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        index = 0
        while 1:
            if index >= len(nums) : break
            if nums[index] == val :
                nums.pop(index)
            else :
                index += 1
        return len(nums)



# 28. 找出字符串中第一个匹配项的下标
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not haystack or not needle : return -1

        index_haystack = 0
        index_needle = 0
        flag = 0
        first_number = 0

        while 1:

            if index_haystack <= len(haystack) and index_needle == len(needle) and flag == 1 : return first_number        # ！
            if index_haystack > len(haystack) - 1 : return -1        # ！
            if haystack[index_haystack] != needle[index_needle] : 
                index_haystack += 1
                if flag == 1 :        # ！
                    flag = 0
                    index_haystack = first_number + 1        # ！
                    first_number = 0
                    index_needle = 0


            elif haystack[index_haystack] == needle[index_needle] : 

                if index_needle == 0 :         # ！
                    flag = 1
                    first_number = index_haystack
                index_haystack += 1        # ！
                index_needle += 1




# 29. 两数相除

# 自己的超出时间限制
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        
        abs_dividend = abs(dividend)
        abs_divisor = abs(divisor)

        if abs_dividend < abs_divisor or abs_dividend == 0: return 0
        if abs_divisor == 1 : 
            if dividend >= 0 and divisor <=0 or dividend <= 0 and divisor >=0 :
                result = - abs_dividend
                return -2 ** 31  if result < -2 ** 31 else result

            else : 
                result = abs_dividend
                return 2 ** 31 -1  if result > 2 ** 31 - 1 else result
        
        abs_result = 0
        number = 0
        while abs_dividend >= abs_result :
            abs_result += abs_divisor
            number += 1
        
        if dividend >= 0 and divisor <=0 or dividend <= 0 and divisor >=0 :
            result = - (number - 1)
            return -2 ** 31  if result < -2 ** 31 else result

        else : 
            result = number - 1
            return 2 ** 31 -1  if result > 2 ** 31 - 1 else result





# 二分查找 + 二分快速加法
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        
        INI_MIN, INT_MAX = -2 ** 31, 2 ** 31 -1
        if dividend == INI_MIN :
            if divisor == 1 : return INI_MIN
            elif divisor == -1 : return INT_MAX
        if dividend == 0 : return 0
        if divisor == INI_MIN : return 1 if dividend == INI_MIN else 0     # ！

        flag = False
        if dividend > 0 :
            dividend = - dividend     # ！
            flag = not flag     # ！不能 -flag  变成 0
        if divisor > 0 :
            divisor = - divisor
            flag = not flag

        def quickAdd(divisor2, factor, dividend2) :     # ！
            accumulator = 0
            add = divisor2
            while factor :
                if (factor & 1) == 1 :     # ！奇数
                    if accumulator < dividend2 - add :
                        return False
                    accumulator += add
                if factor != 1 :     # ！偶数
                    if add < dividend2 - add :
                        return False
                    add += add
                factor >>= 1      # ！除 2
            return True                    

        left, right, result = 1, INT_MAX, 0
        while left <= right :      # ！
            mid = left + ((right - left) >> 1)     # ！运算顺序 先加法后移位
            is_bigger_than_divident = quickAdd(divisor, mid, dividend)
            if is_bigger_than_divident :       # ！
                result = mid      # ！写在这里保证// 去余数取整.  之前不写，导致if else都能有各自mid 被用作result
                left = mid + 1
                if left > INT_MAX :       # ！ >=
                    break
            else :
                right = mid - 1
        return -result if flag else result





# 30. 串联所有单词的子串
# 自己的，超出时间限制
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        index_list = []
        for s_index in range(len(s) - len(words) * len(words[0]) + 1) :      # ！
            
            words_copy = copy.deepcopy(words)
            length = len(words_copy)
            is_match = True

            for number in range(length) :
                if s[s_index + len(words[0]) * number : s_index + len(words[0]) * (number + 1)] in words_copy :     # ！
                    words_copy.remove(s[s_index + len(words[0]) * number : s_index + len(words[0]) * (number + 1)])     # ！

                else :
                    is_match = False
                    break
            if is_match : index_list.append(s_index)

        return index_list

# 官方答案，滑窗法。放入Counter()特殊的字典，方便加减，通过字典记录和匹对
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        index_list = []
        length_s, length_words, length_one_word = len(s), len(words), len(words[0])       # ！
        for index_one_word in range(length_one_word) :       # ！
            if index_one_word + length_one_word * length_words > length_s :       # ！
                break
            accumulator = Counter()       # ！
            for index_words in range(length_words) :
                word = s[index_one_word + index_words * length_one_word : \
                        index_one_word + (index_words + 1) * length_one_word]
                accumulator[word] += 1
            for index_words in range(length_words) :
                word = words[index_words]
                accumulator[word] -= 1       # ！永远的负值
                if accumulator[word] == 0 :
                    del accumulator[word]
            for index in range(index_one_word, length_s - length_one_word * length_words + 1, length_one_word) :
                if index != index_one_word :       # ！！第二次自己写和 0 比较。每一次index_one_word的最开始都要跳过
                    word = s[index + (length_words - 1) * length_one_word : \
                            index + length_words * length_one_word]       # ！删前取后，仅在s中
                    accumulator[word] += 1
                    if accumulator[word] == 0 :       # ！！第二次自己写，漏，word前后并不一样
                        del accumulator[word]
                    word = s[index - length_one_word : index]       # ！删前取后
                    accumulator[word] -= 1
                    if accumulator[word] == 0 :
                        del accumulator[word]

                if len(accumulator) == 0 :
                    index_list.append(index)


        return index_list


# 31. 下一个排列
# 一开始还以为，做数值列表，二分查找，二分排序
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        
        for index in range(len(nums)) :
            right = len(nums) - index - 1
            left = right - 1 
            if left >= 0 :
                if right == len(nums) - 1 :
                    if nums[left] < nums[right] :
                        nums[left], nums[right] = nums[right], nums[left]
                        break
                if nums[left] >= nums[right] :       # ！
                    continue
                else :
                    exchange_index = right       # ！
                    for index_inner in range(right + 1, len(nums)) :
                        if nums[index_inner] > nums[left]:
                            exchange_index = index_inner       # ！没有 break
                        elif nums[index_inner] <= nums[left] : break
                    nums[left], nums[exchange_index] = nums[exchange_index], nums[left]       # ！
                    
                    nums_2_sort = nums[right:]       # ！ 不能直接 nums[right:].sort()
                    nums_2_sort.sort()
                    nums[right:] = nums_2_sort       # ！不能 对 nums 做一些操作。官方无法直接识别 + 
                    break
            else : nums.sort()




# 32. 最长有效括号
# 双指针 左到右 右到左
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        
        if len(s) < 2 : return 0
        max_len = 0
        length = 0

        left, right = 0, 0
        for move in range(len(s)) :
            if s[move] == '(' : left += 1
            elif s[move] == ')' : right += 1
            if right == left and right * 2 > max_len: max_len = right * 2       # ！ right * 2 > max_len 漏乘 2
            elif right > left : left, right = 0, 0
        left, right = 0, 0      # ！
        for move in range(len(s) - 1, -1, -1) :      # ！
            if s[move] == '(' : left += 1
            elif s[move] == ')' : right += 1
            if right == left and right * 2 > max_len: max_len = right * 2
            elif right < left : left, right = 0, 0
        return max_len




# 自己写，逻辑不通，无法通过。想list.pop，记录max_len和index，想回头查找不符合的源头，从左到右，有些情况没法判断
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        
        if len(s) < 2 : return 0
        max_len = 0
        max_len_start_index = 0 
        record_list = []
        length = 0

        for move in range(len(s)) :
            if s[move] == '(' :
                record_list.append(move)

            elif s[move] == ')' :
                
                if len(record_list) == 0 :
                    continue
                
                max_len_start_index = record_list[-1]
                record_list.pop()
                length += 2

        if len(record_list) != 0 :
        return max_len



# 33. 搜索旋转排序数组

class Solution:
    def search(self, nums: List[int], target: int) -> int:

        def find_2(left2, right2, target2) :
            while left2 < right2 :     # ！
                mid2 = (left2 + right2) // 2
                if nums[mid2] == target2 : return mid2
                elif nums[left2] == target2 : return left2
                elif nums[right2] == target2 : return right2     # ！
                elif nums[mid2] < target2 : 
                    left2 = mid2 + 1
                elif nums[mid2] > target2 :      # ！
                    right2 = mid2 - 1
            return -1


        left = 0 
        right = len(nums) - 1
        while left <= right :     # ！
            mid = (left + right) // 2
            
            if nums[mid] == target : return mid
            
            if nums[left] <= nums[mid] and nums[left] <= target and target <= nums[mid] :     # ！
                return find_2(left, mid, target)

            elif nums[mid] <= nums[right] and nums[mid] <= target and target <= nums[right] :     # ！
                return find_2(mid, right, target)

            elif nums[left] > nums[mid] :     # ！
                right = mid - 1

            elif nums[mid] > nums[right] :
                left = mid + 1
            else : return -1     # ！

        return -1

# 官方
class Solution(object):
    def search(self, nums, target):
        if not nums:
            return -1

        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if target == nums[mid]:
                return mid

            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1


# 34. 在排序数组中查找元素的第一个和最后一个位置
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums) - 1
        strat_end = [-1, -1]
        while left <= right :     # ！
            mid = (left + right) // 2
            if nums[mid] == target :
                start = mid
                end = mid
                while 1 :     # ！
                    start -= 1
                    if start >= 0 :     # ！
                        if nums[start] != target:
                            start += 1
                            break
                    else :     # ！
                        start += 1
                        break
                while 1 :
                    end += 1
                    if end <= len(nums) - 1 :     # ！
                        if nums[end] != target:
                            end -= 1
                            break   
                    else :
                        end -= 1
                        break  
                strat_end = [start, end]
                break

            elif nums[mid] > target :
                right = mid - 1
            elif nums[mid] < target :
                left = mid + 1

        return strat_end



# 35. 搜索插入位置

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right :
            mid = (left + right) // 2
            if nums[mid] == target :
                return mid
            elif nums[mid] < target :
                left = mid + 1
            elif nums[mid] > target :
                right = mid - 1
        return left




# 36. 有效的数独
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        
        for index_row in range(len(board)) :
            row_counter = Counter()
            for index_column in range(len(board[0])) :
                if board[index_row][index_column] != '.' :
                    row_counter[board[index_row][index_column]] += 1

            has_duplicates = any(count > 1 for count in row_counter.values())
            if has_duplicates:
                return False

        for index_column in range(len(board[0])) :
            column_counter = Counter()
            for index_row in range(len(board)) :
                if board[index_row][index_column] != '.' :
                    column_counter[board[index_row][index_column]] += 1

            has_duplicates = any(count > 1 for count in column_counter.values())     # ！
            if has_duplicates:
                return False

        for i in range(len(board) // 3) :
            for j in range(len(board) // 3) :     # ！
                nine_counter = Counter()     # ！
                for index_row in range(3 * i, 3 * (i + 1)) :      # ！之前放置反了
                    for index_column in range(3 * j, 3 * (j + 1)) :
                        if board[index_row][index_column] != '.' :
                            nine_counter[board[index_row][index_column]] += 1

                has_duplicates = any(count > 1 for count in nine_counter.values())     # ！
                if has_duplicates:
                    return False

        return True






# 38. 外观数列
class Solution:
    def countAndSay(self, n: int) -> str:
        
        def read_number(number2) :
            left = 0
            result = ''
            for right in range(left + 1, len(number2)) :      # ！
                if number2[left] == number2[right] :
                    continue
                else :
                    result += str(right - left) + number2[left]      # ！多加 1
                    left = right
            
            result += str(right - left + 1) + number2[left]      # ！漏
            return result
        number = ''
        for index in range(n) :
            
            if index == 0 :       # ！
                number = '1'
                
            elif index == 1 : 
                number = '11'
                
            else : number = read_number(number)
        return number





# 39. 组合总和
# 官方
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        path, result, length = [], [], len(candidates)
        def dfs(start, path, target) :                            # ！         
            if target < 0 : return                    # ！ break 和 return 作用相同， 单纯 return 能提前结束运行             
            elif target == 0 :
                result.append(path)                                     # ！
                return
            [dfs(index, path + [candidates[index]], target - candidates[index]) for index in range(start, length)]                     # ！列表推导式需要变成 列表
        dfs(0, path, target)
        return result



class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        def dfs(candidates, begin, size, path, res, target):
            if target == 0:
                res.append(path)
                return

            for index in range(begin, size):
                residue = target - candidates[index]
                if residue < 0:
                    break                    # ！ break 和 return 作用相同， 单纯 return 能提前结束运行

                dfs(candidates, index, size, path + [candidates[index]], res, residue)

        size = len(candidates)
        if size == 0:
            return []
        candidates.sort()
        path = []
        res = []
        dfs(candidates, 0, size, path, res, target)
        return res




# 自己写的，不通过，无法去重，无法找到所有，只能每一个开头找到一个满足的
# 无法找到所有-这个可以解决。   无法去重-这个没法解决
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        
        def dfs(number, candidates2) :
            accumulator = 1
            result_element = []
            for index in range(len(candidates2)) :
                minus = candidates2[index]
                if minus > number :
                    continue
                accumulator = number - minus
                is_leaf_node = flag_pass = False

                if accumulator == 0 :
                    return True, [minus], True
                else : 
                    if is_leaf_node : flag_pass, record_list, is_leaf_node = dfs(accumulator, candidates2[index + 1 :])
                    else : flag_pass, record_list, is_leaf_node = dfs(accumulator,candidates2)

                    if flag_pass :
                        record_list.append(minus)
                        if number == target : result.append(record_list)
                        else : 
                            if not result_element : 
                                result_element.append(record_list)
                            else :
                                a.append(record_list) for a in result_element
                if is_leaf_node : break

            if not result_element : flag_pass = True
            return flag_pass, result_element
   
            if number == target : return result
            if accumulator != 0 : return False,[]
            
            
        dfs(target)
        return result


# 40. 组合总和 II


# 上题思路，别人的去重方法
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path, result, length = [], [], len(candidates)
        candidates.sort()               # ！
        def dfs(start, path, target) :         
            if target < 0 : return                         
            elif target == 0 :
                result.append(path)              # ！ 最关键，最方便
                return              # ！
            for index in range(start, length) :
                if index > start and candidates[index] == candidates[index - 1] : continue               # ！index > 0 导致错误，导致缺少
                dfs(index + 1, path + [candidates[index]], target - candidates[index]) 
        dfs(0, path, target)
        return result


# 剪枝优化  
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path, result, length = [], [], len(candidates)
        candidates.sort()                      # ！
        def dfs(start, path, target) :         
            if target < 0 : return                         
            elif target == 0 :
                result.append(path)
                return
            for index in range(start, length) :
                if index > start and candidates[index] == candidates[index - 1] : continue
                if candidates[index] > target : break                                       # ！
                dfs(index + 1, path + [candidates[index]], target - candidates[index]) 
        dfs(0, path, target)
        return result

# 超出时间限制 
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path, result, length = [], [], len(candidates)
        def dfs(start, path, target) :         
            if target < 0 : return                         
            elif target == 0 :
                path.sort()                 # ！
                if path not in result :                 # ！
                    result.append(path)
                return
            [dfs(index + 1, path + [candidates[index]], target - candidates[index]) for index in range(start, length)]                  # ！
        dfs(0, path, target)
        return result


# 41. 缺失的第一个正数

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n) :
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i] :               # ！不能是 if 。while 保证交换后的另一个仍然满足，否则走过错过
                nums[nums[i] - 1] , nums[i] = nums[i] , nums[nums[i] - 1]
        for i in range(n) :
            if nums[i] != i + 1 :
                return i + 1
        return n + 1

print(Solution().firstMissingPositive([3,4,-1,1]))

# 自己写逻辑不通
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        min_num = 1
        right = False
        flag = 0
        for index in range(len(nums)) :
            if nums[index] < 1 :
                continue
            if nums[index] >= 1 : 
                flag += 1
            if flag == 1 :
                min_num = nums[index]


# 42. 接雨水
# 官方。双指针
class Solution:
    def trap(self, height: List[int]) -> int:
        volumn = left = max_left = max_right = 0
        right = len(height) - 1
        while left < right :
            max_left = max(max_left, height[left])
            max_right = max(max_right, height[right])
            if height[left] < height[right] :
                volumn += max_left - height[left]
                left += 1
            else : 
                volumn += max_right - height[right]
                right -= 1
        return volumn

# 超出时间限制。同时 if  判断太多，屎山
class Solution:
    def trap(self, height: List[int]) -> int:
        volumn = 0
        highest_now = 0
        length = len(height)
        if length <= 2 : return 0
        def find_highest(highest_local, start, end) :
            
            index_highest_local, highest_local = start, height[start]
            for index in range(start, end) :
                if index == start : pass
                elif height[index] >= highest_local : 
                    index_highest_local, highest_local = index, height[index]
                    
            return index_highest_local, highest_local

        
        def calculate_volumn(highest_local, start, end) :
            nonlocal volumn, length
            if start == 0 and end == length : 
                index_highest, highest = find_highest(highest_local, start, end)
                
            flag = True
            if start == 0 :
                if start == end : return
                
                if start == 0 and end == length : 
                    if index_highest == start : flag = not flag
                else : index_highest = end
                
                if flag :     
                    left, highest_2th_l = find_highest(highest_local, 0, index_highest) # find the closest left highest_2th
                    
                    volumn_l = highest_2th_l * (index_highest - left - 1)
                    for j in range(left + 1, index_highest) :
                        volumn_l -= height[j]

                    volumn += volumn_l

                    if left <= 1 and end != length : return
                    
                    calculate_volumn(highest_2th_l, start, left)

            flag = True
            if end == length :
                if start == end : return
                
                if start == 0 and end == length : 
                    if index_highest == end - 1 : flag = not flag
                else : index_highest = start
                
                

                if flag : 
                    index_highest_local = index_highest + 1
                    if index_highest_local > end - 1 : return
                    highest_local = height[index_highest_local]
                    for i in range(index_highest + 1, end) :   # find the closest right highest_2th
                        if i == index_highest + 1 : pass
                        elif height[i] > highest_local : 
                            index_highest_local, highest_local = i, height[i]
                    
                    right, highest_2th_r = index_highest_local, highest_local
                    
                    #if right >= length - 1 and highest_2th_r > height[index_highest + 1] : pass # calculate final right
                    #else : return

                    
                    volumn_r = highest_2th_r * (- index_highest + right - 1)
                    for j in range(index_highest + 1, right) :
                        volumn_r -= height[j]

                    volumn += volumn_r

                    if right >= length - 2 and start != 0 : return
                    calculate_volumn(highest_2th_r, right, end)


        
        
        calculate_volumn(highest_now, 0, length)
        return volumn



#43. 字符串相乘
# 官方
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0' : return '0'                   #！
        number1 = number2 = 0
        m, n = len(num1), len(num2)
        record = [0] * (m + n)                   #！长度估计
        for i in range(len(num1) - 1 , -1, -1) :
            number1 = int(num1[i])
            for j in range(len(num2) - 1 , -1, -1) :
                number2 = int(num2[j])                   #！复制粘贴错误
                record[i + j + 1] += number1 * number2                   #！ +=      =
        for i in range(m + n - 1, 0, -1) :
            record[i - 1] += record[i] // 10
            record[i] = record[i] % 10
        index = 1 if record[0] == 0 else 0                   #！
        result = ''.join(str(x) for x in record[index:])                   #！
        
        return result



# 44. 通配符匹配
# 复习第十题，并反复有实验 * 情况， 认真考虑 i = 0 时候
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1) ]
        dp[0][0] = True

        def matches(i, j):
            if i == 0 : return False
            elif s[i - 1] == p[j - 1] or p[j - 1] == '?' : return True

        for i in range(m + 1):
            for j in  range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] |= dp[i][j - 1]                  #！
                    # if j <= i :dp[i][j] |= dp[j - 1][j - 1]
                    dp[i][j] |= dp[i - 1][j]                              #！简单道理，却断点调试太久
                    

                elif matches(i, j):
                    dp[i][j] |= dp[i - 1][j - 1]
        return dp[-1][-1]  




# 45. 跳跃游戏 II
# 自己写  官方提示  看 两步的收益 now + future
# if max2 <= nums[index] + index - start_init : max2 = nums[index] + index - start_init
class Solution:
    def jump(self, nums: List[int]) -> int:
        
        frequency = 0
        length = len(nums)
        if length == 1 :return 0
        max_jump_length = nums[0]
        start = 0
        flag = 0
        while 1 :
            start_init = start
            if start == 0 : 
                max_jump_length_local = max_jump_length
            end = start_init + max_jump_length_local + 1
            max_one = -1
            max2 = -1
            for index in range(start_init + 1, end):
                if index >= length - 1 :
                    flag = 1
                    break
                if max2 <= nums[index] + index - start_init :
                    #      未来步         已有步  官方答案省去 start_init 并作为列表末尾 判断标志
                    max2 = nums[index] + index - start_init
                    max_one = nums[index]
                    start = index
            frequency += 1
            max_jump_length_local = max_one
            if flag == 1 : break
        
        return frequency


# 官方  抽象，变量少，变量名简洁
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        maxPos, end, step = 0, 0, 0                # 巧妙地初值设定和复用
        for i in range(n - 1):
            if maxPos >= i:
                maxPos = max(maxPos, i + nums[i])
                if i == end:
                    end = maxPos
                    step += 1
        return step


# 断点调试多次
# 自己写的搜索 if max_one <= nums[index] : max_one = nums[index] 无法适应递减序列 [10,9,8,7,6,5,4,3,2,1,1,0]
class Solution:
    def jump(self, nums: List[int]) -> int:
        
        frequency = 0
        length = len(nums)
        if length == 1 :return 0
        max_jump_length = nums[0]
        start = 0
        flag = 0
        while 1 :
            start_init = start
            if start == 0 :                            #！
                max_jump_length_local = max_jump_length
            end = start_init + max_jump_length_local + 1                           #！
            max_one = -1                           #！
            for index in range(start_init + 1, end):                           #！
                if index >= length - 1 :                           #！
                    flag = 1
                    break
                if max_one <= nums[index] :                           #！
                    max_one = nums[index]
                    start = index
            frequency += 1                           #！
            max_jump_length_local = max_one                           #！
            if flag == 1 : break                           #！
        
        return frequency


# 46. 全排列
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        length = len(nums)
        result = []
        def f(depth, little_list) :                          #！
            if depth == length :
                result.append(little_list)
                return
            newlist = list(filter(lambda x: x not in little_list, nums))                          #！
            for x in newlist :                          #！
                if depth == 0 : little_list = []                          #！
                
                f(depth + 1, little_list + [x])                          #！

        
        f(0,[])
        return result

# 官方 动态维护，前处理后还原，更少变量，且不需要filter查找
class Solution:
    def permute(self, nums):
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        res = []
        backtrack()
        return res


# 47. 全排列 II

class Solution:
# 官方 used[]
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:

        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path.copy())
                return
            for i in range(size):
                if not used[i]:

                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue

                    used[i] = True
                    path.append(nums[i])
                    dfs(nums, size, depth + 1, path, used, res)
                    used[i] = False
                    path.pop()

        size = len(nums)
        if size == 0:
            return []

        nums.sort()

        used = [False] * len(nums)
        res = []
        dfs(nums, size, 0, [], used, res)
        return res


# 自己重写官方
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        length = len(nums)
        
        def dfs(depth, path, used):
            if depth == length :
                res.append(path[:])
                return
            for j in range(0, length) :                       #！
                if not used[j] :                       #！used 两个作用 1.遍历 全排列
                    if j > 0 and nums[j] == nums[j - 1] and not used[j - 1]: continue                       #！ 2. 剪枝去重，重复相同，相同但未使用过的，减去 

                    path.append(nums[j])
                    used[j] = True
                    dfs(depth + 1, path, used)
                    used[j] = False                       #！
                    path.pop()                       #！


        nums.sort()                       #！
        used = [False] * length                       #！
        
        dfs(0, [], used)
        return res

# 自己写，15倍时间于原答案
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        def back(first) :
            if first == n :
                if nums not in res :                       #！这样搜索耗时极久
                    res.append(nums[:])                          #！创建副本。但 res.append(nums)  修改原指针位置
                    return
            for i in range(first, n) :
                nums[i], nums[first] = nums[first], nums[i]
                back(first + 1)
                nums[i], nums[first] = nums[first], nums[i]

        back(0)
        return res



# 48. 旋转图像
# 旋转所有圈
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        row = column = len(matrix) - 1
        if row == 1 :
            for j in range(row) :
                matrix[0][j], matrix[j][row], matrix[row][row - j], matrix[row - j][0] = \
                matrix[row - j][0], matrix[0][j], matrix[j][row], matrix[row][row - j]

        for inner_outer in range(row - 1) :

            for j in range(inner_outer, row - inner_outer) :
                matrix[inner_outer][j], matrix[j][row - inner_outer], \
                matrix[row - inner_outer][row - j], matrix[row - j][inner_outer] = \
                \
                matrix[row - j][inner_outer], matrix[inner_outer][j], \
                matrix[j][row - inner_outer], matrix[row - inner_outer][row - j]


# 只旋转最外围
row = column = len(matrix) - 1
for j in range(row) :
    matrix[0][j], matrix[j][row], matrix[row][row - j], matrix[row - j][0] = \
    matrix[row - j][0], matrix[0][j], matrix[j][row], matrix[row][row - j]






# 49. 字母异位词分组
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)                      #！mp是一个字典，其中每个键对应一个空列表
        for word in strs :
            sequence = [0] * 26                      #！
            for char in word :
                sequence[ord(char) - ord('a')] += 1                      #！
            mp[tuple(sequence)].append(word)                      #！ tuple才可哈希，通过26字母位置进行计数

        return list(mp.values())                      #！

# 自己想用set()或集合，集合套集合，键不能说集合，无法哈希，但是不是集合自己就不知道怎么设计区别差异
#。或者用Counter计数，不知道思路



# 50. Pow(x, n)
# 方法同之前的 写出整除的代码。快速幂 + 迭代
class Solution:
    def myPow(self, x: float, n: int) -> float:
        result = 1
        if n < 0 : x, n = 1/x, -n
        while n > 0 :
            if n & 1 == 1 :
                result = result * x
            
            x = x * x
            n = n >> 1
        return result 
# 时间复杂度：O(log⁡n)，即为对 nnn 进行二进制拆分的时间复杂度。
# 空间复杂度：O(1)



# 官方 快速幂 + 递归
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def quickMul(N):
            if N == 0:
                return 1.0
            y = quickMul(N // 2)
            return y * y if N % 2 == 0 else y * y * x
        
        return quickMul(n) if n >= 0 else 1.0 / quickMul(-n)

# 时间复杂度：O(log⁡n)，即为递归的层数。
# 空间复杂度：O(log⁡n)，即为递归的层数。这是由于递归的函数调用会使用栈空间。


# 51. N 皇后

# 官方 由于每个皇后必须位于不同列，因此已经放置的皇后所在的列不能放置别的皇后。第一个皇后有 NNN 列可以选择，
# 第二个皇后最多有 N−1N-1N−1 列可以选择，第三个皇后最多有 N−2N-2N−2 列可以选择
# 方向1 同一条斜线上的每个位置满足行下标与列下标之差相等，例如 (0,0)(0,0)(0,0) 和 (3,3)(3,3)(3,3) 在同一条方向一的斜线上
# 方向二的斜线为从右上到左下方向，同一条斜线上的每个位置满足行下标与列下标之和相等
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def generateBoard():
            board = list()
            for i in range(n):
                row[queens[i]] = "Q"
                board.append("".join(row))
                row[queens[i]] = "."                   #！
            return board

        def backtrack(row: int):
            if row == n:
                board = generateBoard()                   #！
                solutions.append(board)
            else:
                for i in range(n):
                    if i in columns or row - i in diagonal1 or row + i in diagonal2:                   #！
                        continue
                    queens[row] = i                   #！
                    columns.add(i)                   #！
                    diagonal1.add(row - i)                   #！
                    diagonal2.add(row + i)                   #！
                    backtrack(row + 1)
                    columns.remove(i)
                    diagonal1.remove(row - i)
                    diagonal2.remove(row + i)
                    
        solutions = list()
        queens = [-1] * n                   #！
        columns = set()                   #！
        diagonal1 = set()
        diagonal2 = set()
        row = ["."] * n
        backtrack(0)
        return solutions






# 自己写的，用时277ms.久
# 直观的做法是暴力枚举将 NNN 个皇后放置在 N×NN \times NN×N 的棋盘上的所有可能的情况，并对每一种情况判断是否满足皇后彼此之间不相互攻击。
# 暴力枚举的时间复杂度是非常高的，因此必须利用限制条件加以优化。

# 使用 table 记录， 并使用deepcopy , 
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n == 1 : return [['Q']]
        def unaccessable_table(table2, j, k) :                   #！
            flag = False
            for little_j in range(n) :
                table2[little_j][k] = flag
                table2[j][little_j] = flag
                if j + little_j < n and k + little_j < n :                   #！
                    table2[j + little_j][k + little_j] = flag
                if j - little_j >= 0  and k - little_j >= 0 :
                    table2[j - little_j][k - little_j] = flag
                if j + little_j < n and k - little_j >= 0 :
                    table2[j + little_j][k - little_j] = flag
                if j - little_j >= 0  and k + little_j < n :
                    table2[j - little_j][k + little_j] = flag

            return table2


        record = []
        def nqueen(depth, start, table, path) :                   #！
            if depth == n : 
                record.append(path[:])                   #！
                return
            for j in range(start, start + 1) :                   #！

                for k in range(0, n) :                   #！
                    if table[j][k]:                   #！
                        
                        inner_string = ''
                        for inner in range(n):                   #！
                            if inner == k :
                                inner_string += 'Q'
                            else: inner_string += '.'

                        path.append(inner_string)
                        table_copy =  copy.deepcopy(table)                   #！
                        table_copy =  unaccessable_table(table_copy, j, k)

                        nqueen(depth + 1, start + 1, table_copy, path)                   #！

                        path.pop()                   #！

        table = [[True] * n for _ in range(n)]
        
        nqueen(0, 0, table, [])
        return record



# 53. 最大子数组和
# 耗时两个半左右。规则写到人麻
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_list = []
        left = 0
        right = 0
        sum_max = sum_now = sum_last = 0
        while right < len(nums) and left < len(nums):
            if right == 0 : 
                max_list = [0,0]
                sum_now = nums[right]
                sum_max = sum_now
                sum_last = sum_now
            
            
            
            else :  # 考虑 if sum_last
                if nums[right] >= 0 :
                    if sum_last >= 0 :                  #！
                        if sum_max <= sum_last + nums[right] :
                            sum_max = sum_last + nums[right]
                        max_list = [left,right]
                        sum_last = sum_last + nums[right]
                    elif sum_last < 0 :
                        left = right
                        if sum_max <= nums[right] :
                            sum_max = nums[right]
                        max_list = [left,right]
                        sum_last = nums[right]

                elif nums[right] < 0 :                  #！
                    if sum_last <= nums[right] :
                        left = right
                        if sum_max <= nums[right] :
                            sum_max = nums[right]
                        max_list = [left,right]
                        sum_last = nums[right]
                    elif 0 >= sum_last > nums[right] : # 跳过 j 节点                  #！
                        left = right + 1
                        sum_last = 0
                        if left < len(nums) :
                            if sum_max <= sum_last + nums[left] :
                                sum_max = sum_last + nums[left]

                    elif sum_last > 0 :                  #！
                        if sum_last + nums[right] < 0 : # 跳过 j 节点
                            left = right + 1
                            sum_last = 0
                            if left < len(nums) :
                                if sum_max <= sum_last + nums[left] :
                                    sum_max = sum_last + nums[left]
                        elif sum_last + nums[right] >= 0 : # 还是可以又收益
                            sum_last = sum_last + nums[right]

            right += 1
                
        return sum_max
