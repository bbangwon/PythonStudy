# 문제 1의 1부터 n까지의 합 구하기를 재귀 호출로 만들어보세요.
def sum(n):
    if n <= 1:
        return n
    return n + sum(n - 1)

print(sum(100))
