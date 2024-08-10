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

