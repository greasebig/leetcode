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
