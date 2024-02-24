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
时间复杂度是O(n)，  
空间复杂度也是O(n)，因为递归调用会占用栈空间。    

### 尾递归
什么是尾递归呢?(tail recursion), 顾名思议，就是一种“不一样的”递归，说到它的不一样，就得先说说一般的递归。对于一般的递归，比如下面的求阶乘，教科书上会告诉我们，如果这个函数调用的深度太深，很容易会有爆栈的危险。    
原因很多人的都知道，让我们先回顾一下函数调用的大概过程：   
- 调用开始前，调用方（或函数本身）会往栈上压相关的数据，参数，返回地址，局部变量等。
- 执行函数。
- 清理栈上相关的数据，返回。   

因此，在函数 A 执行的时候，如果在第二步中，它又调用了另一个函数 B，B 又调用 C… 栈就会不断地增长不断地装入数据，当这个调用链很深的时候，栈很容易就满了，这就是一般递归函数所容易面临的大问题。     
而尾递归在某些语言的实现上，能避免上述所说的问题，注意是某些语言上，尾递归本身并不能消除函数调用栈过长的问题，那什么是尾递归呢？在上面写的一般递归函数 func() 中，我们可以看到，func(n) 是依赖于 func(n-1) 的，func(n) 只有在得到 func(n-1) 的结果之后，才能计算它自己的返回值，因此理论上，在 func(n-1) 返回之前，func(n)，不能结束返回。因此func(n)就必须保留它在栈上的数据，直到func(n-1)先返回，而尾递归的实现则可以在编译器的帮助下，消除这个限制：   
```python
int tail_func(int n, int res)
{
     if (n <= 1) return res;

     return tail_func(n - 1, n * res);
}
```
从上可以看到尾递归把返回结果放到了调用的参数里。   
这个细小的变化导致，tail_func(n, res)不必像以前一样，非要等到拿到了tail_func(n-1, nres)的返回值，才能计算它自己的返回结果 – 它完全就等于tail_func(n-1, nres)的返回值。    
最终返回结果在传递参数里面    
因此理论上：tail_func(n)在调用tail_func(n-1)前，完全就可以先销毁自己放在栈上的东西。

这就是为什么尾递归如果在得到编译器的帮助下，是完全可以避免爆栈的原因：每一个函数在调用下一个函数之前，都能做到先把当前自己占用的栈给先释放了，尾递归的调用链上可以做到只有一个函数在使用栈，因此可以无限地调用！     
尾递归是一种特殊的递归形式，其特点是函数在最后一步调用自身，而且这个调用不依赖于任何额外的计算。    
可以避免爆栈的原因：每一个函数在调用下一个函数之前，都能做到先把当前自己占用的栈给先释放了，尾递归的调用链上可以做到只有一个函数在使用栈，因此可以无限地调用！

### 尾调用   
前面的讨论一直都集中在尾递归上，这其实有些狭隘，尾递归的优化属于尾调用优化这个大范畴，所谓尾调用，形式它与尾递归很像，   
都是一个函数内最后一个动作是调用下一个函数，不同的只是调用的是谁，显然尾递归只是尾调用的一个特例。     
```python
int func1(int a)
{
   static int b = 3;
   return a + b;
}

int func2(int c)
{
    static int b = 2;
    
    return func1(c+b);
}

```
上面例子中，func2在调用func1之前显然也是可以完全丢掉自己占有的栈空间的，原因与尾递归一样，因此理论上也是可以进行优化的，而事实上这种优化也一直是程序编译优化里的一个常见选项，甚至很多的语言在标准里就直接要求要对尾调用进行优化，原因很明显，尾调用在程序里是经常出现的，优化它不仅能减少栈空间使用，通常也能给程序运行效率带来比较大的提升。   

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
另一方法，看作尾递归
```python
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int n = nums1.length;
    int m = nums2.length;
    int left = (n + m + 1) / 2;
    int right = (n + m + 2) / 2;
    //将偶数和奇数的情况合并，如果是奇数，会求两次同样的 k 。
    return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;  
}
    
    private int getKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        //让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1 
        if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
        if (len1 == 0) return nums2[start2 + k - 1];

        if (k == 1) return Math.min(nums1[start1], nums2[start2]);

        int i = start1 + Math.min(len1, k / 2) - 1;
        int j = start2 + Math.min(len2, k / 2) - 1;

        if (nums1[i] > nums2[j]) {
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        }
        else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }
```  
时间复杂度：每进行一次循环，我们就减少 k/2 个元素，所以时间复杂度是 O(log(k)，而 k=(m+n)/2，所以最终的复杂也就是 O(log(m+n）      
空间复杂度：虽然我们用到了递归，但是可以看到这个递归属于尾递归，所以编译器不需要不停地堆栈，所以空间复杂度为 O(1)   




## 时间复杂度为log
O(log n)的算法通常是二分查找或其他类似的分治算法，  
它们能够通过每一步排除一半的可能性，从而在较短的时间内找到解决方案。这种复杂度通常比线性复杂度（O(n)）更好，尤其在处理大型数据集时表现更为出色。

## 快排/二分查找
- 快排  
pivot，用递归不断大小划分
- 二分查找


## hashmap 列表list
- hashmap = {'I':1, 'V':5, 'X':10}   hashmap记录键值对    
if hashmap[s[i]] >= hashmap[s[i + 1]] :    #! 读hashmap用[] 而不是()   
- 创建列表 f = [[False] * (n + 1) for _ in range(m + 1)]    
- s = [('I', 1), ('V', 5), ('X', 10)] 列表记录键值对    
s = [['I', 1], ['V', 5], ['X', 10]]   这两样读取都正确，读取结果都一样    
for key,value in s   读取为 'I' 和 1   
for key in s   读取为('I', 1)
- new_num_list.append(x2 % 10)    
字符串不可以append，只能 + 或 ''.join()    
char1 + char2 字符串相加    
''.join()的对象是一个列表，而不是输入多个单体    
- 都是用 [] 取值。字符串，列表，hashmap   
函数采用()    
- letter_group + [table[digits[0]]]     # ！少了[]   列表 + 就是concat   
new_combine.append(char1 + char2)       # ！往列表内添加元素，不能list + str     

## 反转链表
```
# reverse none
# 1     - 2      - 3
# head
#      next_node
# reverse
#      head
```

## 深浅拷贝
元组，列表，链表 都需要深拷贝    
reverse2 = reverse    
reverse2.next = None      
reverse2会改变原链表    
如果你想创建一个链表的副本而不影响原始链表，你需要使用深拷贝（deep copy）   
reverse2 = copy.deepcopy(reverse)        

元组original_tuple = ([1, 2, 3], [4, 5, 6])   
列表original_list = [[1, 2, 3], [4, 5, 6]]

