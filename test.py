import sys

'''
c= input()
a=c.split()

n = int(c)
for _ in range(n):
    ls = list(map(int, input().split()))
    print(ls[0] + ls[1])
'''
'''
a = sys.stdin
for b in a:
    pass

for line in sys.stdin: 
    a = line.split() 
    print(int(a[0]) + int(a[1]))
'''

'''
b = list(map(int, "1"))


for line in sys.stdin:
    a = line.split()
    b = list(map(int, a))
    print(sum(b))

'''
#input().split()
'''
def solve() -> None:
    n, m, k = map(int, input().split())
    if k == 1:
        print(m - n)
        return
    ans = 0
    
    while n != m:
        if n // k >= m:
            ans += n % k
            n //= k
        else:
            n -= 1
        ans += 1
    print(ans)    

if __name__ == '__main__':
    solve()
'''

from collections import Counter

'''
def solve() -> None:
    s = input()
    a = input()
    b = input()
    s = [x.lower() for x in s]
    a = [x.lower() for x in a]
    b = [x.lower() for x in b]
    cnts = Counter(s)
    cnta = Counter(a)
    cntb = Counter(b)
    ans = 0
    for i in range(len(s)+1):
        ok = True
        for k, v in cnta.items():
            if cnts[k] < i * v:
                ok = False
        if ok:
            now = len(s)
            for k, v in cntb.items():
                now = min(now, (cnts[k] - i * cnta[k]) // v)
            ans = max(ans, i + now)
    print(ans)

if __name__ == '__main__':
    solve()

'''

'''
def solve() -> None:
    n = int(input())
    adj = [[] for _ in range(n)]
    for i in range(n - 1):
        x, y = map(int, input().split())
        adj[x - 1].append(y - 1)
        adj[y - 1].append(x - 1)
    m = 0
    def dfs(u, fa, c):
        nonlocal m
        if not c:
            m += 1
        for v in adj[u]:
            if v != fa:
                dfs(v, u, not c)
    dfs(0, -1, False)
    print(m * (n - m) - (n - 1))
    

if __name__ == '__main__':
    solve()

'''

'''

# Input: color path and commands
path = input()
command = input()

# Get the lengths of path and command
n = len(path)
m = len(command)

# Initialize the dp table
dp = [[0] * (m + 1) for _ in range(n + 1)]

# Iterate over each character in the path and command
for i in range(1, n + 1):
    for j in range(1, m + 1):
        # If the command matches the current path character
        if command[j - 1] == path[i - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

# Output the length of the longest subarray of identical colors
print(dp[n][m])
'''


'''
gpt4o完败 估计现在出题都后置检查一下gpt？三次不过则算出题成功？
图片不行
gpt4omini更加不行
我的逻辑也理不清




import sys

path = input()
command = input()

m = len(path)
n = len(command)

dp = [[0]*(n+1) for _ in range(m+1)]


for i in range(1, m + 1):
    for j in range(1, n + 1):
        if command[j - 1] == '*':
            dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        elif path[i - 1] == command[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = dp[i][j - 1]

print(dp[m][n])



for i in range(1, m + 1):
    for j in range(1, n + 1):
        if command[j - 1] == '*':
            dp[i][j] = max(dp[i][j - 1]+1, dp[i - 1][j])  # *可以匹配0次或多次
        elif path[i - 1] == command[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1  # 匹配当前字符
        else:
            dp[i][j] = dp[i][j - 1]  # 尝试不匹配当前字符






for i in range(n+1):
    for j in range(m+1):
        if j > 0 and command[j-1] == '*':
            if i>1:
                if path[i-1] == path[i-2]:
                    dp[i][j] = dp[i-1][j-1] + 1 
            
        
        if i>0 and j>0 and command[j-1] == path[i-1]:
            dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1)

'''

'''
N = int(input())
blk = int(input())
nums = [int(c) for c in input().split()]

res = {}

for i in range(0, N, blk):
    cur_blk = nums[i:i + blk]
    tp = tuple(cur_blk)
    if tp not in res:
        res[tp] = [i, 1]  # 下标、出现次数
    else:
        res[tp][1] += 1

sorted_res = sorted(res.items(), key=lambda x: x[1][0])
ans = []
for key, value in sorted_res:
    ans.extend(key)  # 将整个数据块加入结果
    ans.append(value[1])  # 加入计数

print(' '.join(map(str, ans)))
'''

'''

INF = 20220201

def main():
    import sys
    #input = sys.stdin.read
    data = []
    index = 0

    def next_int():
        nonlocal index
        index += 1
        return int(data[index-1])

    n, m, a = 3,6,2
    graph = [[] for _ in range(n + 1)]
    for _ in range(3):
        vi, ui, wi = 1,2,1
        graph[vi].append((ui, wi))
    for _ in range(3):
        vi, ui, wi = 2,3,1
        graph[vi].append((ui, wi))

    dp = [[0] * (a + 1) for _ in range(n + 1)]
    st = [[False] * (a + 1) for _ in range(n + 1)]
    dp[1][0] = 1

    for i in range(a):
        for v in range(1, n + 1):
            if dp[v][i] > 0 or st[v][i]:
                for u, w in graph[v]:
                    if i + w <= a:
                        dp[u][i + w] = (dp[u][i + w] + dp[v][i]) % INF
                        if dp[u][i + w] + dp[v][i] >= INF:
                            st[u][i + w] = True
                        if st[v][i]:
                            st[u][i + w] = True

    if st[n][a]:
        print("All roads lead to Home!")
    print(dp[n][a])

if __name__ == "__main__":
    main()

'''
'''
a=[1,2,3]
print('a=[1,2,3]')
a.reverse
print('a.reverse  ',a)
a=[1,2,3]
print('a=[1,2,3]')
a.reverse() 
print('a.reverse()  ',a)
a=3e-5
print('a=3e-5 ',a)
print('a*10 ',a*10)
print('a*100 ',a*100)
#print(a.2f)
'''

'''
import sys

n,k = map(int,input().split(' '))
a= list(map(int,input().split(' ')))

win_rate=0

for i in range(n):
    #per_win_rate += a[i]/(a[i]+a[j])
    per_win_rate=1
    # k=1
    for j in range(n):
        if j==i : continue
        per_win_rate *= a[i]/(a[i]+a[j])
    win_rate += per_win_rate
print(f'{win_rate:.2f}')

'''
'''
import sys

n=int(input())
string_a=input()

stack_a = []
length=0
count=0
for i in range(n):
    if string_a[i]=='(' :
        stack_a.append(string_a[i])
        length+=1
        count+=1
    else:
        if count<=0: break
        if stack_a.pop()=='(':
            length+=1
            count-=1

print(length)
'''


# 这道题好像就是得用递归或者dfs写
'''
import sys

n,k = map(int,input().split(' '))
a= list(map(int,input().split(' ')))

win_rate=0

per_win_rate = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        per_win_rate[i][j] = a[i]/(a[i]+a[j])
win_1_rate=[]

for i in range(n):
    #if k==1
    temp=1
    for j in range(n):
        if i==j: continue
        temp *= per_win_rate[i][j]
    #if k=2:
    # win_rate += temp
    win_1_rate.append(temp)

for i in range(n):
    if k==1:
        win_rate+=win_1_rate[i]
    if k==2:
        for j in range(n):
            if j==i: continue
            win_rate+=win_1_rate[i]*win_1_rate[j]
    if k==3:
        for j in range(n):
            if j==i: continue
            for k in range(n):
                if k==i or k==j: continue
                win_rate+=win_1_rate[i]*win_1_rate[j]*win_1_rate[k]


print(f'{win_rate:.2f}')


'''


'''
import re

def process_string(s):
    # 先处理所有的大小写转换

    # 定义正则表达式模式，匹配 #(upper)[...] 的结构
    pattern = r"#\(upper\)\[([^\]]+)\]"

    # 使用re.sub进行替换，匹配到的内容用lambda函数来转换为大写
    output_str = re.sub(pattern, lambda match: f"{match.group(1).upper()}", s)

    # 定义正则表达式模式，匹配 #(upper)[...] 的结构
    pattern = r"#\(lower\)\[([^\]]+)\]"

    # 使用re.sub进行替换，匹配到的内容用lambda函数来转换为大写
    output_str = re.sub(pattern, lambda match: f"{match.group(1).lower()}", output_str)

    s = output_str
    
    result = ""
    i = 0
    while i < len(s):
        if s[i].isdigit():
            # 获取重复次数
            count = int(s[i])
            i += 1
            # 处理括号内的内容
            if s[i] == '[':
                j = i + 1
                depth = 1
                while depth > 0:
                    if s[j] == '[':
                        depth += 1
                    elif s[j] == ']':
                        depth -= 1
                    j += 1
                sub_result = process_string(s[i+1:j-1])
                result += sub_result * count
                i = j
            else:
                # 处理单个字符的重复
                result += s[i] * count
                i += 1
        else:
            # 处理普通字符
            result += s[i]
            i += 1
    return result

def apply_functions(s):
    # 处理upper、lower等函数
    # 这里可以扩展，支持更多的函数
    return process_string(s)
    

# 示例用法
s1 = "3[a]2[bc]"
s2 = "3[#(upper)[a]2[bc]]"
#print(apply_functions(s1))  # 输出 aaabcbc
#print(apply_functions(s2))  # 输出 AbcbcAbcbcAbcbc
s1 = "3[[a]2#(upper)[bc]]"
print(apply_functions(s1))  # 输出 aaabcbc

'''
'''
import sys

def main(n,nums):

  nums2 = nums.copy()  # Create a copy for sorting
  nums2.sort()

  dic = {}  # Dictionary to store index mapping
  for i, num in enumerate(nums2):
    dic[num] = i

  s = 0
  for i, num in enumerate(nums):
    if i % 2 != dic[num] % 2:
      s += 1

  print(s // 2)

if __name__ == "__main__":
  main(4,[2,4,3,1])
  main(4,[1,2,3,4])
'''

'''

print([1]*2)
print([[1]*2 for i in range(3)])
print([[1 for _ in range(2)] for i in range(3)])
print([[1]*2]*3)
'''


'''


first_line = list('QWERTYUIOP')
print(first_line)

first_line = ''.join(first_line)
#first_line = str(first_line)
print(first_line)


first_line = 'QWERTYUIOP'
#for i in first_line:
    #print(i)

'''

'''
from collections import defaultdict  
  
T = int(input())  
mp = defaultdict(int)  # 使用defaultdict来自动处理不存在的键  
  
while T > 0:  
    T -= 1  
  
    x, y = map(int, input().split())  
    l = max((key for key in mp if key <= x), default=None)  
  
    # 向右移动x直到找到一个空位或超出mp的范围  
    while l is not None and mp[l] >= x:  
        x = mp[l] + 1  
        l = max((key for key in mp if key <= x), default=None)  
  
    r = min((key for key in mp if key > x), default=None)  
    # 清理x到x+y-1之间的所有已占用位置  
    if r is not None:  
        while r is not None and r <= x + y - 1:  
            temp = mp[r]  
            del mp[r]  
            y += temp - r + 1  
            r = min((key for key in mp if key > r), default=None)  
  
    mp[x] = x + y - 1  
    print(x + y - 1)

'''

'''
a = [9,8,7]
sorteda = sorted(range(len(a)),key=lambda x:a[x])
print(sorteda)
'''



'''
from collections import deque
maze = eval(input())
direction = [(0,-1),(0,1),(-1,0),(1,0)]
#print(maze)
start_point = []
end_point = []
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i][j] == 'S':
            start_point = (i,j)
        elif maze[i][j] == 'E':
            end_point = (i,j)

rows, cols = len(maze), len(maze[0])
single_direction = ['D', 'U', 'L', 'R']
visited = [[False]*len(maze[0])   for _ in range(len(maze))]


def bfs(maze, start, end):
    queue = deque([(start, 1)])
    while queue:
        position, path_length = queue.popleft()
        x,y = position
        visited[x][y] == True
        if position == end:
            return path_length
        for (dx,dy) in direction:
            if maze[x][y] in single_direction:
                index = single_direction.index(maze[x][y])
                if (dx,dy) != direction[index]: continue
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[x][y] != '1' and not visited[nx][ny]:
                queue.append(((nx,ny), path_length + 1))
    return -1
                
                
    
print(bfs(maze, start_point, end_point))    
'''
    
    
    
    
'''  
class Solution:
    def assignJobs(self , n: int, x: int, y: int, z: int) -> int:
        # write code here
        
        count = 0
        def dfs(n1,x1,y1,z1):
            nonlocal count
            if n1 == 0:
                count += 1
                return 
            if x1-1 >=0 : dfs(n1-1, x1-1, y1, z1)
            if y1-1 >=0 : dfs(n1-1, x1, y1-1, z1)
            if z1-1 >=0 : dfs(n1-1, x1, y1, z1-1)

        dfs(n,x,y,z)
        return count

print(Solution().assignJobs(5,1,2,3))
'''

'''
class Solution:
    def assignJobs(self , n: int, x: int, y: int, z: int) -> int:
        # write code here
        
        current = [0,0,0]
        def dfs(n1,x1,y1,z1,current,index):
            if current[0] > x1 or current[1] > y1 or current[2] > z1:
                return 0
            if index == n1:
                return 1
            total_ways = 0
            for i in range(3):
                current[i] += 1
                total_ways += dfs(n1, x1, y1, z1, current, index+1)
                current[i] -= 1
            return total_ways
        return dfs(n,x,y,z,current,0)
print(Solution().assignJobs(5,1,2,3))

'''


'''
class Solution:
    def assignJobs(self , n: int, x: int, y: int, z: int) -> int:
        # write code here
        
        count = 0
        for  i in range(min(n,x)+1):
            for j in range(min(n-i,y)+1):
                k = n-i-j
                if k<=z and k>= 0:
                    count += 1
        return count
        
print(Solution().assignJobs(5,1,2,3))
'''


x_list=[1,2,3,4]
y_list=[5,6,7,8]
place = [(x,y) for x,y in zip(x_list, y_list)]
print(place)
print(zip(x_list, y_list))
x,y = place[0]
print(x,y)
place_set = set(place)
print(len(place_set))
for item in place_set:
    print(item)














