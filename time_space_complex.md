# 时间复杂度与空间复杂度
## leetcode 3 for循环和字典
```python
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
```
该算法的时间复杂度是 O(n)，空间复杂度是 O(n)
- 时间复杂度  
循环遍历字符串一次，使用两个指针（left_pointer 和 right_pointer）来维护无重复字符子串的窗口。在每次迭代中，对字符进行检查和记录。由于每个字符只会被处理一次，因此总的时间复杂度是 O(n)，其中 n 是输入字符串的长度。
- 空间复杂度  
使用了一个字典 char_record_map 来记录每个字符最后一次出现的位置。在最坏的情况下，所有字符都不重复，此时字典的大小会达到字符串的长度。因此，空间复杂度是 O(n)，其中 n 是输入字符串的长度。额外使用了几个整数变量，但它们的空间占用是常量级别的，不会随着输入规模增加而变化。

## 递归算法 计算阶乘 n!
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```
时间复杂度是O(n)，空间复杂度也是O(n)，因为递归调用会占用栈空间。  

## leetcode 3 递归 
举例，如果递归函数中调用两次递归函数，则时间复杂度是O(2^n)，如下：  
使用了递归调用栈，空间复杂度主要取决于递归调用的深度。在最坏的情况下，递归深度为 n，因此空间复杂度为 O(n)。    
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        return self._lengthOfLongestSubstring(s, 0, set())

    def _lengthOfLongestSubstring(self, s: str, start: int, char_set: set) -> int:
        if start == len(s):
            return 0
        without_current_char = self._lengthOfLongestSubstring(s, start + 1, char_set)
        if s[start] not in char_set:
            char_set.add(s[start])
            with_current_char = 1 + self._lengthOfLongestSubstring(s, start + 1, char_set)
            char_set.remove(s[start])
            return max(without_current_char, with_current_char)
        return without_current_char
```
