# 1부터 n까지 연속한 숫자의 제곱의 합을 구하는 프로그램을 for 반복문으로 만들어 보세요.
# 예를 들어 n = 10이라면 1^2 + 2^2 + 3^3 + ... + 10^2 = 385

def func(n):
    s = 0
    for i in range(1, n+1):
        s = s + (i ** 2)
    return s

print(func(10))